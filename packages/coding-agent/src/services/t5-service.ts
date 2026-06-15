/**
 * Lightweight T5 grammar correction service using Transformers.js.
 *
 * Runs a small T5 grammar correction model (rabden/t5-tiny-gec-hone, ~11MB
 * quantized) directly in Node.js via ONNX Runtime. No Python, no subprocess,
 * no external API calls.
 *
 * The model is loaded lazily on first use and cached in
 * ~/.cache/huggingface for subsequent runs.
 */

let pipelineModule: typeof import("@huggingface/transformers") | null = null;
let generator:
	| ((text: string, options?: Record<string, unknown>) => Promise<Array<{ generated_text: string }>>)
	| null = null;
let loadPromise: Promise<void> | null = null;

/**
 * Lazy-load the T5 grammar correction model.
 *
 * Downloads on first call (cached to ~/.cache/huggingface thereafter).
 * Subsequent calls reuse the loaded model.
 */
async function ensureLoaded(): Promise<void> {
	if (generator) return;
	if (loadPromise) return loadPromise;

	loadPromise = (async () => {
		try {
			pipelineModule = await import("@huggingface/transformers");
			const { pipeline } = pipelineModule;
			// Use a tiny grammar correction model fine-tuned for fixing
			// subject-verb agreement, tense, articles, casing, etc.
			// ~11MB quantized, ~30-115ms inference on CPU.
			const pipe = await pipeline("text2text-generation", "rabden/t5-tiny-gec-hone", {
				quantized: true,
				dtype: "q8",
			});
			generator = pipe;
		} catch (err) {
			loadPromise = null;
			throw err;
		}
	})();

	return loadPromise;
}

/**
 * Run STT text through the T5 grammar correction model.
 *
 * Fixes grammar errors the regex pass might miss — subject-verb agreement,
 * tense consistency, articles, sentence casing.
 *
 * Falls back to the input text on any error (model not loaded, timeout, etc.).
 */
export async function correctGrammar(text: string): Promise<string> {
	// Skip very short fragments — not worth the inference cost
	if (text.length < 10) return text;

	try {
		await ensureLoaded();
		if (!generator) return text;

		const result = await generator(text, {
			max_new_tokens: 128,
			temperature: 0,
			do_sample: false,
		});

		const corrected = result?.[0]?.generated_text;
		return corrected?.trim() || text;
	} catch {
		// Graceful degradation — return original text
		return text;
	}
}

/**
 * Check if the T5 grammar correction model is available.
 * Useful for determining whether to show the "grammar correction" option
 * in settings or the UI.
 */
export async function isGrammarModelAvailable(): Promise<boolean> {
	try {
		await ensureLoaded();
		return generator !== null;
	} catch {
		return false;
	}
}
