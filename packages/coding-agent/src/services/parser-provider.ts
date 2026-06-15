/**
 * Parser provider interface and config.
 *
 * Parsers clean up raw STT output — fixing transcription errors, resolving
 * self-corrections, and normalizing filler words — before passing the text
 * to the agent.
 */

export interface ParserConfig {
	/** Enable the parser step */
	enabled: boolean;
	/** Ollama-compatible API endpoint (e.g. http://localhost:11434/v1) */
	endpoint: string;
	/** Model name to use (e.g. qwen2.5:1.5b) */
	model: string;
	/** Per-parse timeout in milliseconds */
	timeoutMs: number;
}

/** Default parser config (matches Ollama defaults) */
export const DEFAULT_PARSER_CONFIG: ParserConfig = {
	enabled: true,
	endpoint: "http://localhost:11434/v1",
	model: "qwen2.5:1.5b",
	timeoutMs: 3000,
};

export interface ParserProvider {
	readonly config: ParserConfig;

	/**
	 * Clean *rawText* and return the parsed version.
	 * On timeout/error, returns the raw text unchanged (graceful degradation).
	 */
	parse(rawText: string, signal?: AbortSignal): Promise<string>;
}
