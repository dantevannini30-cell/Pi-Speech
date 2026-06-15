/**
 * TTS polisher provider and regex-based implementation.
 *
 * Polishes agent output text before sending it to the TTS engine, removing
 * markdown formatting, code blocks, thinking tags, file paths, and other
 * artifacts that sound unnatural when read aloud.
 *
 * Uses only deterministic regex — no LLM calls.
 */

export interface TTSPolisherConfig {
	/** Enable the polisher step */
	enabled: boolean;
}

/** Default TTS polisher config (polisher enabled by default) */
export const DEFAULT_TTS_POLISHER_CONFIG: TTSPolisherConfig = {
	enabled: true,
};

export interface TTSPolisherProvider {
	readonly config: TTSPolisherConfig;

	/**
	 * Polish *rawText* for TTS output and return the cleaned version.
	 * Always synchronous — no network calls.
	 */
	polish(rawText: string): Promise<string>;
}

/**
 * Strip thinking/scratchpad tags and their content.
 * Matches  ... ,  ... , and similar markers.
 */
function stripThinkingTags(text: string): string {
	return text.replace(/<(?:think|scratchpad|reasoning|thought|planning|reflection)>[\s\S]*?<\/\1>/gi, "");
}

/**
 * Remove fenced code blocks (``` ... ```) and indented code blocks.
 * Replaces with a brief "code block" summary so the listener knows
 * something was omitted.
 */
function stripCodeBlocks(text: string): string {
	// Fenced code blocks
	text = text.replace(/```[\s\S]*?```/g, " code snippet ");
	// Indented code blocks (4+ spaces or a tab)
	text = text.replace(/^(?: {4,}|\t).*/gm, "");
	return text;
}

/**
 * Remove markdown formatting while keeping readable content.
 */
function stripMarkdown(text: string): string {
	// Remove images: ![alt](url)
	text = text.replace(/!\[([^\]]*)\]\([^)]*\)/g, "");

	// Convert links: [text](url) -> text
	text = text.replace(/\[([^\]]*)\]\([^)]*\)/g, "$1");

	// Remove inline code backticks, keep the content
	text = text.replace(/`([^`]+)`/g, "$1");

	// Remove markdown headings markers but keep the heading text
	text = text.replace(/^#{1,6}\s+/gm, "");

	// Remove bold/italic markers
	text = text.replace(/\*{1,3}([^*]+)\*{1,3}/g, "$1");
	text = text.replace(/_{1,3}([^_]+)_{1,3}/g, "$1");

	// Remove horizontal rules (lines that are only ---, ***, ___)
	text = text.replace(/^[-*_]{3,}\s*$/gm, "");

	// Remove blockquote markers
	text = text.replace(/^>\s*/gm, "");

	// Remove table rows and separators (lines starting with | or containing |---|)
	text = text.replace(/^\|.*\|$/gm, "");
	text = text.replace(/^[-:| ]+\|[-:| ]+$/gm, "");
	text = text.replace(/^[-:| ]+$/gm, "");

	return text;
}

/**
 * Clean up list markers and diff syntax.
 */
function stripListAndDiffMarkers(text: string): string {
	// Unordered list markers: keep the content
	text = text.replace(/^[\s]*[-*+]\s+/gm, "");
	// Ordered list markers: keep the content
	text = text.replace(/^[\s]*\d+\.\s+/gm, "");
	// Checklist items: - [ ] or - [x]
	text = text.replace(/^[\s]*[-*+]\s*\[[ x]\]\s*/gm, "");
	// Diff lines (git-style +, - prefixes)
	text = text.replace(/^[+-]\s/gm, "");
	// Diff @@ headers
	text = text.replace(/^@@[\s\S]*?@@$/gm, "");
	return text;
}

/**
 * Replace kebab-case, snake_case, and file paths with space-separated words.
 * This makes filenames like "my-file-name" or "my_file_name" readable as
 * "my file name" rather than the TTS reading the punctuation literally.
 */
function humanizeIdentifiers(text: string): string {
	// Replace path separators with " slash " so /home/user/file reads naturally
	text = text.replace(/\//g, " slash ");

	// Replace underscores between words with spaces (snake_case -> words)
	text = text.replace(/([a-zA-Z0-9])_([a-zA-Z0-9])/g, "$1 $2");

	// Replace hyphens between lowercase sequences (kebab-case -> words)
	// This avoids replacing hyphens used as dashes in sentences
	text = text.replace(/([a-z])-([a-z])/g, "$1 $2");
	text = text.replace(/([a-z])-([0-9])/g, "$1 $2");
	text = text.replace(/([0-9])-([a-z])/g, "$1 $2");

	// Separate camelCase into words: "myFile" -> "my File"
	text = text.replace(/([a-z])([A-Z])/g, "$1 $2");

	// Replace dots between words with " dot " for version numbers, file extensions
	// E.g., "file.txt" -> "file dot txt"
	text = text.replace(/([a-zA-Z])\.([a-zA-Z])/g, "$1 dot $2");

	return text;
}

/**
 * Normalize punctuation for natural speech.
 */
function normalizePunctuation(text: string): string {
	// Replace em-dashes with " — " (spoken as a pause)
	text = text.replace(/—/g, " — ");
	// Replace en-dashes with " to "
	text = text.replace(/–/g, " to ");
	// Remove decorative characters
	text = text.replace(/→/g, " to ");
	text = text.replace(/←/g, " to ");
	text = text.replace(/⇒/g, " to ");
	// Remove arrow sequences like ->, =>, <-, <=
	text = text.replace(/[=]=>/g, "");
	text = text.replace(/[-]=>/g, "");
	text = text.replace(/<=[=]?/g, "");
	// Remove triple dots ... and replace with single …
	text = text.replace(/\.{3,}/g, "…");
	// Ensure sentences end with a period if they don't have terminal punctuation
	text = text.replace(/(\w)$/gm, "$1.");
	// Remove repeated punctuation like !!!, ??, etc.
	text = text.replace(/([!?]){2,}/g, "$1");
	// Remove HTML entities
	text = text.replace(/&amp;/g, "&");
	text = text.replace(/&lt;/g, "<");
	text = text.replace(/&gt;/g, ">");
	text = text.replace(/&quot;/g, '"');
	text = text.replace(/&#39;/g, "'");
	return text;
}

/**
 * Clean up whitespace — collapse multiple spaces and newlines.
 */
function cleanWhitespace(text: string): string {
	// Collapse multiple spaces into one
	text = text.replace(/[ ]{2,}/g, " ");
	// Remove spaces before punctuation
	text = text.replace(/\s+([.,!?;:])/g, "$1");
	// Collapse multiple newlines into at most one
	text = text.replace(/\n{3,}/g, "\n\n");
	// Trim each line
	const lines = text.split("\n").map((l) => l.trim());
	text = lines.join("\n");
	return text.trim();
}

/**
 * Regex-based TTS polisher.
 *
 * Strips markdown, code, thinking tags, file paths, and formatting
 * artifacts from text before it's sent to the TTS engine.
 *
 * Fully deterministic — no LLM calls, no network, no latency.
 */
export class RegexTTSPolisher implements TTSPolisherProvider {
	readonly config: TTSPolisherConfig;

	constructor(config: TTSPolisherConfig) {
		this.config = config;
	}

	async polish(rawText: string): Promise<string> {
		// Skip polishing very short text — likely a brief acknowledgement
		// like "Sure!" or "Done." with no formatting to clean.
		if (rawText.length < 30) {
			return rawText;
		}

		let text = rawText;

		// Order matters — strip structural elements first
		text = stripThinkingTags(text);
		text = stripCodeBlocks(text);
		text = stripMarkdown(text);
		text = stripListAndDiffMarkers(text);
		text = humanizeIdentifiers(text);
		text = normalizePunctuation(text);
		text = cleanWhitespace(text);

		return text;
	}
}
