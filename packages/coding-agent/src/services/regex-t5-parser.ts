/**
 * STT parser that uses regex preprocessing + optional T5 grammar correction.
 *
 * Replaces the earlier Ollama-backed parser — small LLMs kept answering
 * questions instead of cleaning text. This uses deterministic regex for
 * filler removal, self-correction resolution, and homophone fixes, with
 * an optional T5 grammar correction pass for harder cases.
 *
 * The T5 pass runs via Transformers.js (rabden/t5-tiny-gec-hone, ~11MB
 * quantized, ~30-115ms on CPU) and gracefully degrades if unavailable.
 */

import type { ParserConfig, ParserProvider } from "./parser-provider.ts";
import { correctGrammar } from "./t5-service.ts";

// ─── Filler word removal ───────────────────────────────────────────────────

/** Set of filler words to remove entirely. */
const FILLER_WORDS = new Set([
	"um",
	"uh",
	"er",
	"ah",
	"hmm",
	"mm",
	"mhm",
	"uhhuh",
	"uh-huh",
	"uh_uh",
	"umm",
	"erm",
	"eh",
]);

/** Filler phrases (multi-word) to remove. */
const FILLER_PHRASES = [/\byou know\b/gi, /\bi mean\b/gi, /\byou know what i mean\b/gi, /\bi guess\b/gi];

/**
 * Remove filler words from transcribed speech.
 * 'like' is kept when it's a qualifier (e.g. "that file, like the one in docs")
 * but removed when it's used as a pure filler (e.g. "I was like going to the store").
 * We err on the side of keeping 'like' since it's often semantic.
 */
function removeFillers(text: string): string {
	// Remove multi-word filler phrases
	for (const pattern of FILLER_PHRASES) {
		text = text.replace(pattern, "");
	}

	// Remove single-word fillers (word boundary matched)
	// Use a regex that matches fillers as standalone words
	const fillerPattern = new RegExp(`\\b(${Array.from(FILLER_WORDS).join("|")})\\b`, "gi");
	text = text.replace(fillerPattern, "");

	return text;
}

// ─── Self-correction resolution ────────────────────────────────────────────

/**
 * Resolve self-corrections — when the speaker changes their mind mid-sentence.
 *
 * Patterns:
 *   "wait no X" -> drop everything before X, keep X
 *   "actually no X" -> drop everything before X, keep X
 *   "scratch that Y" -> drop everything before Y, keep Y
 *   "forget that Z" -> drop everything before Z, keep Z
 *   "ignore that Z" -> drop everything before Z, keep Z
 *   "I mean Z" -> drop everything before Z, keep Z
 *   "or rather Z" -> drop everything before Z, keep Z
 *   "actually Z" -> when Z is the correction, keep Z
 *   "no wait Z" -> drop everything before Z, keep Z
 */
function resolveSelfCorrections(text: string): string {
	// These markers indicate the speaker is abandoning what they just said
	const correctionMarkers = [
		/\bwait no\b/i,
		/\bno wait\b/i,
		/\bwait\b/i,
		/\bactually no\b/i,
		/\bscratch that\b/i,
		/\bforget that\b/i,
		/\bforget it\b/i,
		/\bignore that\b/i,
		/\bnever mind\b/i,
		/\bnah\b/i,
		/\bnaw\b/i,
	];

	// Process from right to left so we don't interfere with earlier corrections
	for (const marker of correctionMarkers) {
		const match = text.match(marker);
		if (match && match.index !== undefined) {
			// Find the start of the sentence/segment containing the marker
			const beforeMarker = text.slice(0, match.index);
			const sentenceStart =
				Math.max(
					beforeMarker.lastIndexOf(". "),
					beforeMarker.lastIndexOf("! "),
					beforeMarker.lastIndexOf("? "),
					beforeMarker.lastIndexOf("\n"),
				) + 2;

			// Keep everything from the marker onward (the correction)
			text = text.slice(0, sentenceStart > 1 ? sentenceStart : 0) + text.slice(match.index + match[0].length);
		}
	}

	// Handle "I mean" specially — it connects to the correction after it
	text = text.replace(/\bI mean\b/gi, "");

	// Handle "or rather"
	text = text.replace(/\bor rather\b/gi, "");

	return text;
}

// ─── Homophone correction ──────────────────────────────────────────────────

// Words that indicate a verb/contraction context (not a possessive).
const THEYRE_TRIGGERS = new Set([
	"going",
	"coming",
	"running",
	"trying",
	"doing",
	"making",
	"taking",
	"getting",
	"saying",
	"asking",
	"telling",
	"working",
	"all",
	"not",
	"really",
	"very",
	"so",
	"just",
	"already",
	"still",
	"also",
	"here",
	"there",
	"now",
]);

const YOURE_TRIGGERS = new Set([
	"going",
	"coming",
	"running",
	"trying",
	"doing",
	"making",
	"taking",
	"getting",
	"saying",
	"asking",
	"telling",
	"working",
	"a",
	"an",
	"the",
	"so",
	"very",
	"really",
	"not",
	"just",
	"right",
	"wrong",
	"correct",
	"amazing",
	"awesome",
	"great",
	"welcome",
	"all",
	"already",
	"still",
]);

/**
 * Fix common STT homophone errors deterministically.
 */
function fixHomophones(text: string): string {
	// "their are" / "Their are" -> "they're" / "They're"
	text = text.replace(/\b(T?)heir are\b/gi, (_, cap: string) => `${cap || ""}hey're`);
	// "your are" / "Your are" -> "you're" / "You're"
	text = text.replace(/\b(Y?)our are\b/gi, (_, cap: string) => `${cap || ""}ou're`);

	// "their" + trigger word -> "they're"
	text = text.replace(/\b(T?)heir\s+(?=\w+)/gi, (match, cap: string) => {
		const rest = match.slice(match.indexOf(" ") + 1);
		const firstWord = rest.split(/\s/)[0].toLowerCase();
		if (THEYRE_TRIGGERS.has(firstWord)) {
			return `${cap || ""}hey're `;
		}
		return match;
	});

	// "your" + trigger word -> "you're"
	text = text.replace(/\b(Y?)our\s+(?=\w+)/gi, (match, cap: string) => {
		const rest = match.slice(match.indexOf(" ") + 1);
		const firstWord = rest.split(/\s/)[0].toLowerCase();
		if (YOURE_TRIGGERS.has(firstWord)) {
			return `${cap || ""}ou're `;
		}
		return match;
	});

	// its/it's: "its going" -> "it's going"
	text = text.replace(
		/\bits\s+(going|coming|running|trying|doing|making|been|not|really|very|so|just|already|still|here|there|now)\b/gi,
		"it's $1",
	);

	// "thats" -> "that's", "whats" -> "what's", etc.
	text = text.replace(/\b(thats|whats|hows|wheres|whys|whens)\b/gi, (match) => {
		const map: Record<string, string> = {
			thats: "that's",
			whats: "what's",
			hows: "how's",
			wheres: "where's",
			whys: "why's",
			whens: "when's",
			Thats: "That's",
			Whats: "What's",
			Hows: "How's",
			Wheres: "Where's",
			Whys: "Why's",
			Whens: "When's",
		};
		return map[match] || match;
	});

	// Run-together words
	text = text.replace(/\bwassup\b/gi, "what's up");
	text = text.replace(/\bgonna\b/gi, "going to");
	text = text.replace(/\bwanna\b/gi, "want to");
	text = text.replace(/\bgotta\b/gi, "got to");
	text = text.replace(/\bdunno\b/gi, "don't know");
	text = text.replace(/\bcuz\b/gi, "because");
	text = text.replace(/\bcos\b/gi, "because");
	text = text.replace(/\b'em\b/gi, "them");

	// to/too/two: sentence-start "To" -> "Too" when it means "also"
	// This is hard to do perfectly with regex, handle the obvious cases
	text = text.replace(/\bto (much|many|bad|good|quick|slow|fast|late|early)\b/gi, (match) =>
		match.replace(/^to /i, "too "),
	);

	return text;
}

// ─── Punctuation and casing normalization ───────────────────────────────────

/**
 * Normalize punctuation and sentence casing.
 */
function normalizePunctuation(text: string): string {
	// Ensure sentence ends with period if no terminal punctuation
	text = text.replace(/(\w)$/gm, "$1.");
	// Capitalize first letter of sentences
	text = text.replace(/(?:^|(?:[.!?]\s+))([a-z])/g, (match) => match.toUpperCase());
	// Remove spaces before punctuation
	text = text.replace(/\s+([.,!?;:])/g, "$1");
	// Ensure single space after punctuation
	text = text.replace(/([.,!?;:])(?!\s)(\w)/g, "$1 $2");
	// Remove repeated punctuation
	text = text.replace(/([!?]){2,}/g, "$1");
	// Collapse multiple spaces
	text = text.replace(/[ ]{2,}/g, " ");
	return text.trim();
}

// ─── Main parser ───────────────────────────────────────────────────────────

/**
 * Parser that uses regex preprocessing + optional T5 grammar correction.
 *
 * 1. Regex pass (always): remove fillers, resolve self-corrections,
 *    fix homophones, normalize punctuation
 * 2. T5 grammar pass (optional): fix remaining grammar issues via
 *    Transformers.js
 */
export class RegexT5Parser implements ParserProvider {
	readonly config: ParserConfig;

	constructor(config: ParserConfig) {
		this.config = config;
	}

	async parse(rawText: string, _signal?: AbortSignal): Promise<string> {
		if (!rawText || rawText.trim().length === 0) {
			return rawText;
		}

		// Step 1: Regex preprocessing (always runs)
		let text = rawText;

		// Collapse multiple spaces first to make pattern matching easier
		text = text.replace(/[ ]{2,}/g, " ");

		text = resolveSelfCorrections(text);
		text = removeFillers(text);
		text = fixHomophones(text);
		text = normalizePunctuation(text);

		// Clean up any leftover whitespace from removed text
		text = text.replace(/[ ]{2,}/g, " ").trim();

		// Step 2: T5 grammar correction (optional, graceful degradation)
		if (this.config.enabled && text.length >= 10) {
			try {
				text = await correctGrammar(text);
			} catch {
				// Graceful degradation — return regex-cleaned text
			}
		}

		return text;
	}
}
