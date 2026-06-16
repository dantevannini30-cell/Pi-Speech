/**
 * STT (Speech-to-Text) service for Whisper-Pi.
 *
 * Wraps the TextStream ASR client so the TUI can start/stop dictation
 * and receive streaming transcription in real time.
 *
 * Optionally integrates a ParserProvider to clean up raw transcription
 * before emitting the onTranscription callback.
 *
 * When STT is disabled, this service is a no-op and all methods return
 * immediately with empty/null results.
 */

import type { ParserProvider } from "./parser-provider.ts";
import { TextStreamClient, type TextStreamClientConfig, type TextStreamLoadMode } from "./textstream-client.ts";

export interface STTServiceConfig {
	enabled: boolean;
	engine?: string; // kept for backward compatibility, unused
	loadMode?: TextStreamLoadMode;
	debug?: boolean; // when true, emit verbose debug logs to stderr
	noVad?: boolean; // when true, disable Silero VAD (feed all audio to ASR)
}

export type STTState = "idle" | "recording" | "transcribing" | "parsing";

/**
 * Callbacks emitted by the STT service.
 */
export interface STTCallbacks {
	onStateChange?: (state: STTState) => void;
	onTranscription?: (text: string) => void;
	onInterimTranscription?: (text: string) => void;
	onError?: (error: Error) => void;
}

/**
 * Speech-to-text service that uses TextStream (Qwen3-ASR) for streaming dictation.
 *
 * Provides streaming interim transcriptions via onInterimTranscription and final
 * transcriptions via onTranscription.
 *
 * If a ParserProvider is configured, the final transcription text is run through
 * the parser before being emitted via onTranscription(). The "parsing" state
 * is set while the parser runs.
 */
export class STTService {
	private client: TextStreamClient | null = null;
	private config: STTServiceConfig;
	private callbacks: STTCallbacks;
	private _state: STTState = "idle";
	private parserProvider: ParserProvider | null = null;
	private parserEnabled: boolean = true;
	private debug: boolean = false;
	private finalizedBuffer: string = "";
	private lastDraftText: string = "";

	constructor(config: STTServiceConfig, callbacks: STTCallbacks = {}) {
		this.config = config;
		this.debug = config.debug ?? false;
		this.callbacks = callbacks;
		if (config.enabled) {
			this.createClient(config);
		}
	}

	/** Log only when debug is enabled. */
	private log(...args: unknown[]): void {
		if (this.debug) console.debug(...args);
	}

	get state(): STTState {
		return this._state;
	}

	get enabled(): boolean {
		return this.config.enabled;
	}

	get loadMode(): TextStreamLoadMode {
		return this.client?.loadModeSetting ?? "auto";
	}

	/**
	 * Enable or disable the STT service.
	 * When disabled, all methods become no-ops.
	 */
	setEnabled(enabled: boolean): void {
		this.config.enabled = enabled;
		if (enabled && !this.client) {
			this.createClient(this.config);
		} else if (!enabled && this.client) {
			this.client.stop().catch(() => {});
			this.client = null;
		}
	}

	/**
	 * Set the parser provider used to clean transcription output.
	 * Pass null to disable parsing entirely.
	 */
	setParser(provider: ParserProvider | null, enabled?: boolean): void {
		this.parserProvider = provider;
		if (enabled !== undefined) {
			this.parserEnabled = enabled;
		}
	}

	/**
	 * Enable or disable Silero VAD at runtime.
	 * When disabled, all audio (including silence) is fed to the ASR model.
	 * Calls GET /vad?enabled=true|false on the textstream server.
	 */
	async setVadEnabled(enabled: boolean): Promise<boolean> {
		if (!this.client || !this.config.enabled) return false;
		return this.client.setVadEnabled(enabled);
	}

	/** Enable or disable the parser step. */
	setParserEnabled(enabled: boolean): void {
		this.parserEnabled = enabled;
	}

	/** Check whether the parser step is enabled. */
	getParserEnabled(): boolean {
		return this.parserEnabled && this.parserProvider !== null;
	}

	/**
	 * Set the TextStream load mode.
	 */
	setLoadMode(mode: TextStreamLoadMode): void {
		this.config.loadMode = mode;
		if (this.client) {
			this.client.setLoadMode(mode);
		}
	}

	/**
	 * Create the TextStream client and wire its callbacks.
	 */
	private createClient(config: STTServiceConfig): void {
		const clientConfig: TextStreamClientConfig = {
			debug: config.debug ?? false,
			noVad: config.noVad ?? false,
			loadMode: config.loadMode ?? "auto",
		};
		this.client = new TextStreamClient(clientConfig);

		this.client.onFinalizedText = (text) => {
			// Server sends full stable text each time, so replace (not append)
			this.log("[STT] onFinalizedText:", JSON.stringify(text.slice(-80)));
			this.finalizedBuffer = text;
			this.callbacks.onInterimTranscription?.(this.finalizedBuffer);
		};

		this.client.onDraftText = (text) => {
			this.log("[STT] onDraftText:", JSON.stringify(text.slice(-60)));
			// Store last draft as fallback for when /engine drain misses it
			if (text) {
				this.lastDraftText = text;
			}
			// Draft text can be shown as interim if we have a draft callback
			if (this.client && this.client.state === "connected") {
				// Interim transcription shows the draft appended to finalized buffer
				this.callbacks.onInterimTranscription?.(this.finalizedBuffer + text);
			}
		};

		this.client.onError = (error) => {
			this.handleError(error);
		};

		this.client.onConnected = () => {
			this.log("[STT] SSE connected");
		};

		this.client.onDisconnected = () => {
			this.log("[STT] SSE disconnected");
		};

		// For "auto" load mode, start the server immediately so it's ready
		// for the first push-to-talk. Errors are routed via onError.
		if (config.loadMode !== "lazy") {
			this.client.start().catch(() => {});
		}
	}

	/**
	 * Ensure the TextStream server is running.
	 * Respects the load mode setting: in "auto" mode, the server was already
	 * started in the constructor. In "lazy" mode, start is deferred to
	 * startRecording().
	 */
	async ensureRunning(): Promise<boolean> {
		if (!this.client || !this.config.enabled) return false;
		return this.client.start();
	}

	/**
	 * Start recording (push-to-talk).
	 * Resumes the TextStream mic capture and starts receiving SSE events.
	 */
	async startRecording(): Promise<string | null> {
		if (!this.client || !this.config.enabled) {
			this.log("[STT] startRecording: client or config disabled");
			return null;
		}

		this.finalizedBuffer = "";
		this.lastDraftText = "";
		this.setState("recording");
		this.log("[STT] startRecording: calling client.resume()");

		const ok = await this.client.resume();
		if (!ok) {
			this.log("[STT] startRecording: client.resume() returned false");
			this.handleError(new Error("TextStream failed to resume"));
			this.setState("idle");
			return null;
		}

		this.log("[STT] startRecording: resumed OK, returning streaming");
		return "streaming";
	}

	/**
	 * Stop recording and get the final transcription.
	 * Pauses the TextStream mic capture, drains any remaining draft text,
	 * and runs the accumulated finalized text through the parser.
	 */
	async stopRecording(): Promise<void> {
		if (!this.client || !this.config.enabled) {
			this.log("[STT] stopRecording: client or config disabled");
			return;
		}

		this.setState("transcribing");

		// Pause and drain any remaining draft
		this.log("[STT] stopRecording: calling client.pause()");
		const draftText = await this.client.pause();
		this.log("[STT] stopRecording: draftText from pause:", JSON.stringify(draftText));

		// Combine finalized buffer with any drained draft text.
		// Fall back to lastDraftText (from SSE) when the /engine drain misses it
		// due to the race between SSE broadcast and /engine poll.
		let combined = this.finalizedBuffer;
		const effectiveDraft = draftText || this.lastDraftText;
		if (effectiveDraft) {
			combined = combined ? combined + " " + effectiveDraft : effectiveDraft;
		}
		this.log(
			"[STT] stopRecording: finalizedBuffer:",
			JSON.stringify(this.finalizedBuffer),
			"draftText:",
			JSON.stringify(draftText),
			"lastDraftText:",
			JSON.stringify(this.lastDraftText),
			"combined:",
			JSON.stringify(combined),
		);
		this.finalizedBuffer = "";
		this.lastDraftText = "";

		this.setState("idle");

		if (!combined) {
			this.log("[STT] stopRecording: combined empty, returning early");
			return;
		}

		// Run through parser if configured and enabled
		const text = this.parserEnabled && this.parserProvider ? await this.runParser(combined) : combined;
		this.log("[STT] stopRecording: final text:", JSON.stringify(text));

		if (text) {
			this.log("[STT] stopRecording: firing onTranscription");
			this.callbacks.onTranscription?.(text);
		} else {
			this.log("[STT] stopRecording: text empty after parser");
		}
	}

	/**
	 * Run the raw text through the parser with state tracking.
	 * Falls back to raw text on any error.
	 */
	private async runParser(rawText: string): Promise<string> {
		if (!this.parserProvider) return rawText;

		this.setState("parsing");
		try {
			return await this.parserProvider.parse(rawText);
		} catch {
			// Graceful degradation — return raw text
			return rawText;
		} finally {
			this.setState("idle");
		}
	}

	/** Abort current recording without transcribing and reset state to idle. */
	abort(): void {
		this.finalizedBuffer = "";
		this.lastDraftText = "";
		this.client?.pause().catch(() => {});
		this.setState("idle");
	}

	/** Get the current accumulated finalized text (for display purposes). */
	getCurrentFinalizedText(): string {
		return this.finalizedBuffer;
	}

	private setState(state: STTState): void {
		this._state = state;
		this.callbacks.onStateChange?.(state);
	}

	private handleError(error: Error): void {
		this.callbacks.onError?.(error);
	}
}
