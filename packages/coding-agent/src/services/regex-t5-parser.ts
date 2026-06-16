/**
 * STT parser that uses T5 grammar correction.
 *
 * Runs transcribed text through Google's T5 model (rabden/t5-tiny-gec-hone,
 * ~11MB quantized, ~30-115ms on CPU) for grammar correction.
 * Gracefully degrades if the model is unavailable.
 */

import type { ParserConfig, ParserProvider } from "./parser-provider.ts";
import { correctGrammar } from "./t5-service.ts";

// ─── Main parser ───────────────────────────────────────────────────────────

/**
 * Parser that runs T5 grammar correction on transcribed STT text.
 * No regex preprocessing — raw transcription is sent directly to T5.
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

		let text = rawText.trim();

		// T5 grammar correction (optional, graceful degradation)
		if (this.config.enabled && text.length >= 10) {
			try {
				text = await correctGrammar(text);
			} catch {
				// Graceful degradation — return raw text
			}
		}

		return text;
	}
}
