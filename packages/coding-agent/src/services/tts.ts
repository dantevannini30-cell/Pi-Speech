/**
 * TTS (Text-to-Speech) service for Whisper-Pi.
 *
 * Provides speech synthesis by:
 * 1. Calling the Python whisper_bot TTS CLI (Piper/Kokoro) — primary
 * 2. Falling back to macOS `say` command if Python/piper unavailable
 *
 * The Python CLI wrappers are in packages/coding-agent/whisper-bot/:
 *   - tts_cli.py           — one-shot TTS (speak full text)
 *   - tts_streaming_cli.py — streaming TTS (sentences via stdin)
 */

import { type ChildProcess, spawn } from "node:child_process";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

/** Get the absolute path to the whisper-bot CLI scripts directory. */
function getWhisperBotDir(): string {
	// __dirname equivalent in ESM
	const __filename = fileURLToPath(import.meta.url);
	const __dirname = path.dirname(__filename);
	// We're in packages/coding-agent/dist/services/tts.js
	// The whisper-bot dir is at packages/coding-agent/whisper-bot/
	return path.resolve(__dirname, "../../whisper_bot");
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface TTSServiceConfig {
	enabled: boolean;
	provider?: "piper" | "kokoro"; // which TTS engine to use
	voice?: string; // voice name (e.g. "en_US-lessac-medium" for Piper, "af_heart" for Kokoro)
}

export type TTSState = "idle" | "speaking";

export interface TTSCallbacks {
	onStateChange?: (state: TTSState) => void;
	onSentenceStart?: (sentence: string) => void;
	onSentenceEnd?: (sentence: string) => void;
	onDone?: () => void;
	onError?: (error: Error) => void;
}

// ---------------------------------------------------------------------------
// Service
// ---------------------------------------------------------------------------

/**
 * Text-to-speech service.
 *
 * Uses the Python whisper_bot TTS CLI by default, with macOS `say` as
 * a fallback.
 */
export class TTSService {
	private config: TTSServiceConfig;
	private callbacks: TTSCallbacks;
	private _state: TTSState = "idle";
	private whisperBotDir: string;
	private currentProcess: ChildProcess | null = null;
	private childProcesses: ChildProcess[] = [];
	private queue: string[] = [];
	private isProcessing = false;

	constructor(config: TTSServiceConfig, callbacks: TTSCallbacks = {}) {
		this.config = config;
		this.callbacks = callbacks;
		this.whisperBotDir = getWhisperBotDir();
	}

	get state(): TTSState {
		return this._state;
	}

	get enabled(): boolean {
		return this.config.enabled;
	}

	get provider(): string {
		return this.config.provider ?? "piper";
	}

	get voice(): string | undefined {
		return this.config.voice;
	}

	/**
	 * Enable or disable TTS.
	 * When disabled, speak() and speakSentence() are no-ops.
	 */
	setEnabled(enabled: boolean): void {
		this.config.enabled = enabled;
		if (!enabled) {
			this.stop();
		}
	}

	/**
	 * Set the voice.
	 */
	setVoice(voice: string): void {
		this.config.voice = voice;
	}

	/**
	 * Speak a full text. Queues it and processes the queue.
	 */
	async speak(text: string): Promise<void> {
		if (!this.config.enabled || !text.trim()) return;

		this.queue.push(text);
		if (!this.isProcessing) {
			await this.processQueue();
		}
	}

	/**
	 * Speak a single sentence immediately. Used for streaming TTS
	 * where sentences arrive one at a time from the agent stream.
	 */
	async speakSentence(sentence: string): Promise<void> {
		if (!this.config.enabled || !sentence.trim()) return;

		if (this.isProcessing) {
			this.queue.push(sentence);
		} else {
			await this.executeWithFallback(sentence, true);
		}
	}

	/**
	 * Immediately stop all speech.
	 */
	stop(): void {
		for (const proc of this.childProcesses) {
			try {
				proc.kill("SIGTERM");
			} catch {
				/* ignore */
			}
		}
		if (this.currentProcess) {
			try {
				this.currentProcess.kill("SIGTERM");
			} catch {
				/* ignore */
			}
		}
		this.currentProcess = null;
		this.childProcesses = [];
		this.queue = [];
		this.isProcessing = false;
		this.setState("idle");
	}

	// ------------------------------------------------------------------
	// Queue processing
	// ------------------------------------------------------------------

	private async processQueue(): Promise<void> {
		this.isProcessing = true;

		while (this.queue.length > 0) {
			const text = this.queue.shift()!;
			await this.executeWithFallback(text, false);
		}

		this.isProcessing = false;
		this.setState("idle");
		this.callbacks.onDone?.();
	}

	// ------------------------------------------------------------------
	// Python TTS (Piper/Kokoro)
	// ------------------------------------------------------------------

	/**
	 * Speak text via the Python one-shot CLI.
	 */
	private async executePythonOneShot(text: string): Promise<void> {
		return new Promise((resolve, reject) => {
			this.setState("speaking");
			this.callbacks.onSentenceStart?.(text);

			const args = [path.join(this.whisperBotDir, "tts_cli.py"), "--text", text, "--provider", this.provider];
			if (this.config.voice) {
				args.push("--voice", this.config.voice);
			}

			const proc = spawn("python3", args, {
				cwd: path.dirname(this.whisperBotDir),
				stdio: ["ignore", "pipe", "pipe"],
			});
			this.currentProcess = proc;
			this.childProcesses.push(proc);

			proc.on("error", (err) => {
				this.callbacks.onError?.(err);
				this.cleanupProcess(proc);
				reject(err);
			});

			proc.on("close", (code) => {
				this.cleanupProcess(proc);
				this.callbacks.onSentenceEnd?.(text);
				if (code !== 0) {
					reject(new Error(`Python TTS exited with code ${code}`));
				} else {
					resolve();
				}
			});
		});
	}

	/**
	 * Speak a single sentence via the Python streaming CLI.
	 * The streaming CLI reads sentences from stdin, one per line.
	 */
	private async executePythonStreaming(sentence: string): Promise<void> {
		return new Promise((resolve, reject) => {
			this.setState("speaking");
			this.callbacks.onSentenceStart?.(sentence);

			const args = [path.join(this.whisperBotDir, "tts_streaming_cli.py"), "--provider", this.provider];
			if (this.config.voice) {
				args.push("--voice", this.config.voice);
			}

			const proc = spawn("python3", args, {
				cwd: path.dirname(this.whisperBotDir),
				stdio: ["pipe", "pipe", "pipe"],
			});
			this.currentProcess = proc;
			this.childProcesses.push(proc);

			// Send the sentence and signal EOF to speak it
			proc.stdin?.write(sentence + "\n");
			proc.stdin?.end();

			proc.on("error", (err) => {
				this.callbacks.onError?.(err);
				this.cleanupProcess(proc);
				reject(err);
			});

			proc.on("close", (code) => {
				this.cleanupProcess(proc);
				this.callbacks.onSentenceEnd?.(sentence);
				if (code !== 0) {
					reject(new Error(`Python TTS exited with code ${code}`));
				} else {
					resolve();
				}
			});
		});
	}

	// ------------------------------------------------------------------
	// Fallback: macOS `say`
	// ------------------------------------------------------------------

	/**
	 * Speak text via macOS built-in `say` command (fallback).
	 */
	private async executeSay(text: string): Promise<void> {
		return new Promise((resolve) => {
			this.setState("speaking");
			this.callbacks.onSentenceStart?.(text);

			const args: string[] = [];
			if (this.config.voice) {
				args.push("-v", this.config.voice);
			}
			args.push(text);

			const proc = spawn("say", args);
			this.currentProcess = proc;
			this.childProcesses.push(proc);

			proc.on("error", (err) => {
				this.callbacks.onError?.(err);
				this.cleanupProcess(proc);
				resolve();
			});

			proc.on("close", () => {
				this.cleanupProcess(proc);
				this.callbacks.onSentenceEnd?.(text);
				resolve();
			});
		});
	}

	/**
	 * Execute TTS with Python first, falling back to macOS `say`.
	 * Streaming mode uses the stdin-based CLI; one-shot uses the --text CLI.
	 */
	private async executeWithFallback(text: string, streaming: boolean): Promise<void> {
		// Try Python first
		try {
			if (streaming) {
				await this.executePythonStreaming(text);
			} else {
				await this.executePythonOneShot(text);
			}
			return;
		} catch (err) {
			// Python TTS unavailable, fall back to macOS say
			this.callbacks.onError?.(err instanceof Error ? err : new Error(String(err)));
		}

		// Fallback to macOS say
		await this.executeSay(text);
	}

	// ------------------------------------------------------------------
	// Helpers
	// ------------------------------------------------------------------

	private cleanupProcess(proc: ChildProcess): void {
		if (this.currentProcess === proc) {
			this.currentProcess = null;
		}
		const idx = this.childProcesses.indexOf(proc);
		if (idx !== -1) {
			this.childProcesses.splice(idx, 1);
		}
	}

	private setState(state: TTSState): void {
		this._state = state;
		this.callbacks.onStateChange?.(state);
	}
}
