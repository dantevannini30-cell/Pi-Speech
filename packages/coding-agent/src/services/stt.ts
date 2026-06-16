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
	private finalizedBuffer: string = "";

	constructor(config: STTServiceConfig, callbacks: STTCallbacks = {}) {
		this.config = config;
		this.callbacks = callbacks;
		if (config.enabled) {
			this.createClient(config);
		}
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
			loadMode: config.loadMode ?? "auto",
		};
		this.client = new TextStreamClient(clientConfig);

		this.client.onFinalizedText = (text) => {
			// Accumulate finalized text into buffer
			this.finalizedBuffer += text;
			this.callbacks.onInterimTranscription?.(this.finalizedBuffer);
		};

		this.client.onDraftText = (text) => {
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
			// SSE connected
		};

		this.client.onDisconnected = () => {
			// SSE disconnected
		};
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
		if (!this.client || !this.config.enabled) return null;

		this.finalizedBuffer = "";
		this.setState("recording");

		const ok = await this.client.resume();
		if (!ok) {
			this.handleError(new Error("TextStream failed to resume"));
			this.setState("idle");
			return null;
		}

		return "streaming";
	}

	/**
	 * Stop recording and get the final transcription.
	 * Pauses the TextStream mic capture, drains any remaining draft text,
	 * and runs the accumulated finalized text through the parser.
	 */
	async stopRecording(): Promise<void> {
		if (!this.client || !this.config.enabled) return;

		this.setState("transcribing");

		// Pause and drain any remaining draft
		const draftText = await this.client.pause();

		// Combine finalized buffer with any drained draft text
		let combined = this.finalizedBuffer;
		if (draftText) {
			combined += draftText;
		}
		this.finalizedBuffer = "";

		this.setState("idle");

		if (!combined) {
			return;
		}

		// Run through parser if configured and enabled
		const text = this.parserEnabled && this.parserProvider ? await this.runParser(combined) : combined;

		if (text) {
			this.callbacks.onTranscription?.(text);
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
