/**
 * TTS (Text-to-Speech) service for Whisper-Pi.
 *
 * Keeps a persistent Python subprocess with PiperVoice pre-loaded
 * so model loading happens once, not per sentence. Falls back to
 * macOS `say` if Python/piper is unavailable.
 *
 * The Python worker is at whisper_bot/tts_worker.py and communicates
 * via JSONL on stdin/stdout:
 *   -> {"type":"speak","text":"..."}
 *   <- {"type":"done"}         (playback complete)
 *   -> {"type":"shutdown"}
 */

import { type ChildProcess, spawn } from "node:child_process";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

// ---------------------------------------------------------------------------
// Paths
// ---------------------------------------------------------------------------

function getWhisperBotDir(): string {
	const __filename = fileURLToPath(import.meta.url);
	const __dirname = path.dirname(__filename);
	return path.resolve(__dirname, "../../whisper_bot");
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface TTSServiceConfig {
	enabled: boolean;
	provider?: "piper" | "kokoro";
	voice?: string;
	speed?: number; // playback speed, 0.5-3.0, step 0.25, default 1.0
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

export class TTSService {
	private config: TTSServiceConfig;
	private callbacks: TTSCallbacks;
	private _state: TTSState = "idle";
	private whisperBotDir: string;
	private worker: ChildProcess | null = null;
	private pendingResolve: (() => void) | null = null;
	private pendingReject: ((err: Error) => void) | null = null;
	private buf = "";
	private started = false;

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

	setEnabled(enabled: boolean): void {
		this.config.enabled = enabled;
		if (!enabled) this.stop();
	}

	setVoice(voice: string): void {
		this.config.voice = voice;
	}

	get speed(): number {
		return this.config.speed ?? 1.0;
	}

	setSpeed(speed: number): void {
		this.config.speed = Math.max(0.5, Math.min(3.0, Math.round(speed / 0.25) * 0.25));
	}

	/**
	 * Speak text. Spawns the worker on first call, then reuses it.
	 */
	async speak(text: string): Promise<void> {
		if (!this.config.enabled || !text.trim()) return;
		await this.ensureWorker();
		await this.sendText(text);
	}

	/**
	 * Immediately stop all speech and kill the worker.
	 */
	stop(): void {
		if (this.worker) {
			try {
				this.worker.stdin?.write(JSON.stringify({ type: "shutdown" }) + "\n");
			} catch {
				/* ignore */
			}
			try {
				this.worker.kill("SIGTERM");
			} catch {
				/* ignore */
			}
			this.worker = null;
		}
		this.buf = "";
		this.pendingResolve = null;
		this.pendingReject = null;
		this.started = false;
		this.setState("idle");
	}

	// ------------------------------------------------------------------
	// Worker lifecycle
	// ------------------------------------------------------------------

	private async ensureWorker(): Promise<void> {
		if (this.worker && this.worker.exitCode === null) return;

		const workerScript = path.join(this.whisperBotDir, "tts_worker.py");
		const args = ["--provider", this.provider];
		if (this.config.voice) args.push("--voice", this.config.voice);

		this.worker = spawn("python3", [workerScript, ...args], {
			cwd: path.dirname(this.whisperBotDir),
			stdio: ["pipe", "pipe", "pipe"],
		});

		this.buf = "";
		this.worker.stdout?.setEncoding("utf-8");
		this.worker.stdout?.on("data", (data: string) => {
			this.buf += data;
			this.processResponses();
		});

		this.worker.on("error", (err) => {
			this.callbacks.onError?.(err);
			this.worker = null;
		});

		this.worker.on("close", () => {
			if (this.pendingReject) {
				this.pendingReject(new Error("TTS worker died"));
				this.pendingReject = null;
				this.pendingResolve = null;
			}
			this.worker = null;
			this.started = false;
			this.setState("idle");
		});

		// Wait for worker to signal readiness
		await new Promise<void>((resolve, reject) => {
			const timeout = setTimeout(() => reject(new Error("TTS worker startup timeout")), 15_000);
			const onData = (data: string) => {
				if (data.includes('"ready"')) {
					clearTimeout(timeout);
					this.worker?.stdout?.removeListener("data", onData);
					resolve();
				}
			};
			this.worker?.stdout?.on("data", onData);
			this.worker?.on("close", () => {
				clearTimeout(timeout);
				reject(new Error("TTS worker exited before ready"));
			});
		});

		this.started = true;
	}

	// ------------------------------------------------------------------
	// Send / receive
	// ------------------------------------------------------------------

	private async sendText(text: string): Promise<void> {
		return new Promise((resolve, reject) => {
			if (!this.worker || this.worker.exitCode !== null) {
				reject(new Error("TTS worker not running"));
				return;
			}

			this.pendingResolve = resolve;
			this.pendingReject = reject;

			this.setState("speaking");
			this.callbacks.onSentenceStart?.(text);

			const msg = JSON.stringify({ type: "speak", text, speed: this.speed }) + "\n";
			this.worker.stdin?.write(msg);
		});
	}

	private processResponses(): void {
		const lines = this.buf.split("\n");
		// Keep the last incomplete line in the buffer
		this.buf = lines.pop() ?? "";

		for (const line of lines) {
			if (!line.trim()) continue;
			try {
				const msg = JSON.parse(line);
				if (msg.type === "done") {
					this.callbacks.onSentenceEnd?.(msg.text ?? "");
					const resolve = this.pendingResolve;
					this.pendingResolve = null;
					this.pendingReject = null;
					this.setState("idle");
					resolve?.();
				} else if (msg.type === "error") {
					const reject = this.pendingReject;
					this.pendingResolve = null;
					this.pendingReject = null;
					this.setState("idle");
					reject?.(new Error(msg.message ?? "TTS worker error"));
				}
			} catch {
				// Partial JSON line, wait for more data
			}
		}
	}

	// ------------------------------------------------------------------
	// State
	// ------------------------------------------------------------------

	private setState(state: TTSState): void {
		this._state = state;
		this.callbacks.onStateChange?.(state);
	}
}
