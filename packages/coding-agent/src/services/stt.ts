/**
 * STT (Speech-to-Text) service for Whisper-Pi.
 *
 * Wraps the TypeWhisper HTTP API so the TUI can start/stop dictation
 * and retrieve transcribed text without any Python dependency.
 *
 * When STT is disabled, this service is a no-op and all methods return
 * immediately with empty/null results.
 */

import { TypeWhisperAPI, type TypeWhisperConfig, TypeWhisperError } from "./typewhisper-api.ts";

export interface STTServiceConfig {
	enabled: boolean;
	engine?: string;
}

export type STTState = "idle" | "recording" | "transcribing";

/**
 * Callbacks emitted by the STT service.
 */
export interface STTCallbacks {
	onStateChange?: (state: STTState) => void;
	onTranscription?: (text: string) => void;
	onError?: (error: Error) => void;
}

/**
 * Speech-to-text service that uses TypeWhisper macOS app for dictation.
 * Matches the behavior of whisper-bot's Python TypeWhisperProvider.
 */
export class STTService {
	private api: TypeWhisperAPI | null = null;
	private config: STTServiceConfig;
	private callbacks: STTCallbacks;
	private _state: STTState = "idle";
	private currentSessionId: string | null = null;

	constructor(config: STTServiceConfig, callbacks: STTCallbacks = {}) {
		this.config = config;
		this.callbacks = callbacks;
		if (config.enabled) {
			this.api = new TypeWhisperAPI({ engine: config.engine });
		}
	}

	get state(): STTState {
		return this._state;
	}

	get enabled(): boolean {
		return this.config.enabled;
	}

	/**
	 * Enable or disable the STT service.
	 * When disabled, all methods become no-ops.
	 */
	setEnabled(enabled: boolean): void {
		this.config.enabled = enabled;
		if (enabled && !this.api) {
			this.api = new TypeWhisperAPI({ engine: this.config.engine });
		}
	}

	/**
	 * Check if TypeWhisper is reachable (and auto-launch if needed).
	 * Mirrors Python's _ensure_running().
	 */
	async ensureRunning(): Promise<boolean> {
		if (!this.api || !this.config.enabled) return false;
		return this.api.ensureRunning();
	}

	/**
	 * Start recording (push-to-talk).
	 * First ensures TypeWhisper is running (auto-launches if needed).
	 * Returns the session ID, or null if STT is disabled / failed.
	 */
	async startRecording(): Promise<string | null> {
		if (!this.api || !this.config.enabled) return null;

		// Ensure TypeWhisper is running (matches Python behavior)
		const running = await this.api.ensureRunning();
		if (!running) {
			this.handleError(new Error("TypeWhisper not reachable and could not be auto-launched"));
			return null;
		}

		try {
			const sessionId = await this.api.startDictation();
			this.currentSessionId = sessionId;
			this.setState("recording");
			return sessionId;
		} catch (error) {
			this.handleError(error instanceof Error ? error : new Error(String(error)));
			return null;
		}
	}

	/**
	 * Stop recording and begin transcription.
	 */
	async stopRecording(): Promise<void> {
		if (!this.api || !this.config.enabled || !this.currentSessionId) return;

		try {
			this.setState("transcribing");
			await this.api.stopDictation(this.currentSessionId);
		} catch (error) {
			this.handleError(error instanceof Error ? error : new Error(String(error)));
			this.setState("idle");
		}
	}

	/**
	 * Poll for and return the transcribed text.
	 * Returns empty string if not yet ready or STT is disabled.
	 */
	async getTranscribedText(timeoutMs = 60_000): Promise<string> {
		if (!this.api || !this.config.enabled || !this.currentSessionId) return "";

		try {
			const text = await this.api.waitForTranscription(this.currentSessionId, timeoutMs);
			this.currentSessionId = null;
			this.setState("idle");

			if (text) {
				this.callbacks.onTranscription?.(text);
			}
			return text;
		} catch (error) {
			this.handleError(error instanceof Error ? error : new Error(String(error)));
			this.currentSessionId = null;
			this.setState("idle");
			return "";
		}
	}

	/** Abort current recording without transcribing and reset state to idle. */
	abort(): void {
		this.currentSessionId = null;
		this.setState("idle");
	}

	private setState(state: STTState): void {
		this._state = state;
		this.callbacks.onStateChange?.(state);
	}

	private handleError(error: Error): void {
		this.callbacks.onError?.(error);
	}
}
