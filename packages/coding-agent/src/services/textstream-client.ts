/**
 * TextStream SSE client for streaming STT (speech-to-text).
 *
 * Manages a local TextStream ASR server (Qwen3-ASR 0.6B via MLX on Apple Silicon)
 * as a child process, and maintains a persistent SSE connection for real-time
 * streaming of transcribed text.
 *
 * Features:
 * - Process lifecycle: spawn, health-check, stop, restart with crash recovery
 * - SSE connection for receiving finalized/draft transcription events
 * - PTT gating via HTTP POST /resume and /pause
 * - Drain wait: after pause, polls until draft is empty (200ms timeout)
 * - Load mode: "auto" (start immediately) or "lazy" (defer to first resume)
 * - Crash recovery: 1 auto-restart with debug logging, then error+disable
 */

import { type ChildProcess, spawn } from "node:child_process";
import * as path from "node:path";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_PORT = 7890;
const HEALTH_CHECK_POLL_INTERVAL_MS = 500;
const HEALTH_CHECK_TIMEOUT_MS = 30_000;
const DRAIN_POLL_INTERVAL_MS = 50;
const DRAIN_TIMEOUT_MS = 200;
const MAX_CRASH_RESTARTS = 1;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type TextStreamLoadMode = "auto" | "lazy";

export interface TextStreamClientConfig {
	port?: number;
	loadMode?: TextStreamLoadMode;
	pythonDir?: string; // path to .venv/bin directory
}

export type TextStreamClientState = "stopped" | "starting" | "connected" | "paused" | "disconnected" | "error";

// ---------------------------------------------------------------------------
// SSE Parser
// ---------------------------------------------------------------------------

interface SSEMessage {
	event?: string;
	data?: string;
	id?: string;
}

/**
 * Minimal fetch-based SSE parser for Node.js.
 * Consumes a ReadableStream<Uint8Array> and yields parsed SSE messages.
 */
async function* parseSSE(stream: ReadableStream<Uint8Array>): AsyncGenerator<SSEMessage> {
	const reader = stream.getReader();
	const decoder = new TextDecoder();
	let buffer = "";
	let current: SSEMessage = {};

	try {
		while (true) {
			const { done, value } = await reader.read();
			if (done) break;

			buffer += decoder.decode(value, { stream: true });

			const lines = buffer.split("\n");
			buffer = lines.pop() ?? "";

			for (const line of lines) {
				if (line.startsWith(":")) {
					// Comment line, skip
					continue;
				}

				if (line === "") {
					// Empty line marks end of an event
					if (current.event || current.data !== undefined) {
						yield current;
					}
					current = {};
					continue;
				}

				const colonIndex = line.indexOf(":");
				if (colonIndex === -1) {
					// No colon means the field name is the whole line, value is empty
					const field = line.trim();
					if (field === "event") current.event = "";
					else if (field === "data") current.data = "";
					else if (field === "id") current.id = "";
					continue;
				}

				const field = line.slice(0, colonIndex).trim();
				let value = line.slice(colonIndex + 1);
				if (value.startsWith(" ")) {
					value = value.slice(1);
				}

				if (field === "event") {
					current.event = value;
				} else if (field === "data") {
					current.data = (current.data ?? "") + value;
				} else if (field === "id") {
					current.id = value;
				}
			}
		}
	} finally {
		reader.releaseLock();
	}

	// Yield any remaining partial event
	if (current.event || current.data !== undefined) {
		yield current;
	}
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

/**
 * Client for the TextStream ASR server.
 *
 * Manages a persistent Python process running `textstream --no-browser` and
 * maintains an SSE connection for receiving streaming transcription events.
 */
export class TextStreamClient {
	private port: number;
	private loadMode: TextStreamLoadMode;
	private pythonDir: string;

	private process: ChildProcess | null = null;
	private abortController: AbortController | null = null;
	private _state: TextStreamClientState = "stopped";
	private crashCount: number = 0;
	private started: boolean = false;

	// Callbacks
	onFinalizedText?: (text: string) => void;
	onDraftText?: (text: string) => void;
	onConnected?: () => void;
	onDisconnected?: () => void;
	onStateChange?: (state: TextStreamClientState) => void;
	onError?: (error: Error) => void;

	constructor(config: TextStreamClientConfig = {}) {
		this.port = config.port ?? DEFAULT_PORT;
		this.loadMode = config.loadMode ?? "auto";
		this.pythonDir = config.pythonDir ?? path.join(process.cwd(), "packages/coding-agent/.venv/bin");
	}

	get state(): TextStreamClientState {
		return this._state;
	}

	get isRunning(): boolean {
		return this._state === "connected" || this._state === "paused";
	}

	get loadModeSetting(): TextStreamLoadMode {
		return this.loadMode;
	}

	setLoadMode(mode: TextStreamLoadMode): void {
		this.loadMode = mode;
	}

	// ------------------------------------------------------------------
	// Lifecycle
	// ------------------------------------------------------------------

	/**
	 * Start the TextStream server process.
	 * Spawns `textstream --no-browser` and waits for the health endpoint to respond.
	 * If load mode is "lazy", the process is spawned but SSE connection is deferred.
	 *
	 * If something is already running on the target port, reuses that instance.
	 */
	async start(): Promise<boolean> {
		if (this._state !== "stopped" && this._state !== "error") {
			return true;
		}

		this.setState("starting");

		// First check if something is already running on the port
		const alreadyRunning = await this.healthCheck();
		if (alreadyRunning) {
			this.started = true;
			this.crashCount = 0;
			this.setState("paused");
			this.onConnected?.();
			// Start SSE if load mode is auto
			if (this.loadMode === "auto") {
				this.connectSSE().catch(() => {});
			}
			return true;
		}

		// Spawn the process
		try {
			await this.spawnProcess();
		} catch (error) {
			this.handleError(error instanceof Error ? error : new Error(String(error)));
			return false;
		}

		// Poll for health
		const healthy = await this.waitForHealth();
		if (!healthy) {
			this.handleError(new Error("TextStream failed to become healthy within timeout"));
			return false;
		}

		this.started = true;
		this.crashCount = 0;
		this.setState("paused");
		this.onConnected?.();

		// Start SSE connection if load mode is auto
		if (this.loadMode === "auto") {
			this.connectSSE().catch(() => {});
		}

		return true;
	}

	/**
	 * Stop the TextStream server process.
	 * Sends HTTP /stop request and kills the child process.
	 */
	async stop(): Promise<void> {
		// Disconnect SSE
		this.disconnectSSE();

		// Try graceful shutdown via HTTP
		if (this.started) {
			try {
				await fetch(`http://127.0.0.1:${this.port}/stop`, { method: "POST" });
			} catch {
				// Ignore HTTP errors during shutdown
			}
		}

		// Kill the process
		if (this.process) {
			this.process.kill("SIGTERM");
			// Give it a moment to exit, then force kill
			await new Promise<void>((resolve) => {
				const timeout = setTimeout(() => {
					try {
						this.process?.kill("SIGKILL");
					} catch {
						// Already dead
					}
					resolve();
				}, 2000);

				this.process?.on("exit", () => {
					clearTimeout(timeout);
					resolve();
				});
			});
			this.process = null;
		}

		this.started = false;
		this.crashCount = 0;
		this.setState("stopped");
	}

	/**
	 * Restart the server (stop + start).
	 * Used for crash recovery.
	 */
	async restart(): Promise<boolean> {
		await this.stop();
		return this.start();
	}

	/**
	 * Reset crash count (e.g., after successful manual restart).
	 */
	resetCrashCount(): void {
		this.crashCount = 0;
	}

	// ------------------------------------------------------------------
	// PTT gating
	// ------------------------------------------------------------------

	/**
	 * Resume microphone capture.
	 * If load mode is "lazy" and the server hasn't been started yet, starts it now.
	 * If SSE is not connected, connects it.
	 */
	async resume(): Promise<boolean> {
		if (!this.started) {
			if (this.loadMode === "lazy") {
				const ok = await this.start();
				if (!ok) return false;
			} else {
				return false;
			}
		}

		if (!this.isRunning && this._state !== "starting") {
			// Try to restart
			const ok = await this.restart();
			if (!ok) return false;
		}

		// Connect SSE if not already connected
		if (this._state === "paused" || this._state === "disconnected") {
			this.connectSSE().catch(() => {});
		}

		// Send resume
		try {
			const res = await fetch(`http://127.0.0.1:${this.port}/resume`, {
				method: "POST",
			});
			if (!res.ok) {
				throw new Error(`TextStream /resume failed: ${res.status}`);
			}
			return true;
		} catch (error) {
			this.handleError(error instanceof Error ? error : new Error(String(error)));
			return false;
		}
	}

	/**
	 * Pause microphone capture.
	 * After pausing, waits for any remaining draft text to drain (up to 200ms).
	 * Returns the final draft text if any was captured during the drain wait.
	 */
	async pause(): Promise<string> {
		// Send pause
		try {
			const res = await fetch(`http://127.0.0.1:${this.port}/pause`, {
				method: "POST",
			});
			if (!res.ok) {
				throw new Error(`TextStream /pause failed: ${res.status}`);
			}
		} catch (error) {
			this.handleError(error instanceof Error ? error : new Error(String(error)));
		}

		// Drain wait: poll GET /engine until draft is empty, with 200ms timeout
		const drainText = await this.drainDraft();

		// Disconnect SSE after pause
		this.disconnectSSE();
		this.setState("paused");

		return drainText;
	}

	/**
	 * Poll GET /engine until draft field is empty.
	 * Returns the last non-empty draft text, or empty string.
	 */
	private async drainDraft(): Promise<string> {
		const deadline = Date.now() + DRAIN_TIMEOUT_MS;
		let lastDraft = "";

		while (Date.now() < deadline) {
			try {
				const res = await fetch(`http://127.0.0.1:${this.port}/engine`, {
					method: "GET",
					signal: AbortSignal.timeout(1000),
				});
				if (res.ok) {
					const data = (await res.json()) as Record<string, unknown>;
					const draft = (data.draft as string) ?? "";
					if (!draft) {
						// Draft is empty, we're done
						return lastDraft;
					}
					lastDraft = draft;
				}
			} catch {
				// Ignore poll errors during drain
			}
			await new Promise((r) => setTimeout(r, DRAIN_POLL_INTERVAL_MS));
		}

		return lastDraft;
	}

	// ------------------------------------------------------------------
	// SSE connection
	// ------------------------------------------------------------------

	/**
	 * Connect to the SSE stream endpoint.
	 * Reads events in a loop until the connection is aborted or fails.
	 */
	private async connectSSE(): Promise<void> {
		if (this.abortController) {
			this.abortController.abort();
		}
		this.abortController = new AbortController();

		try {
			const response = await fetch(`http://127.0.0.1:${this.port}/stream`, {
				method: "GET",
				signal: this.abortController.signal,
				headers: {
					Accept: "text/event-stream",
					"Cache-Control": "no-cache",
				},
			});

			if (!response.ok || !response.body) {
				throw new Error(`TextStream SSE connection failed: ${response.status}`);
			}

			this.setState("connected");

			for await (const message of parseSSE(response.body)) {
				if (this.abortController.signal.aborted) break;

				if (message.data) {
					try {
						const payload = JSON.parse(message.data) as Record<string, unknown>;
						const finalized = payload.finalized as string | undefined;
						const draft = payload.draft as string | undefined;

						if (finalized) {
							this.onFinalizedText?.(finalized);
						}
						if (draft !== undefined && draft !== null) {
							this.onDraftText?.(String(draft));
						}
					} catch {
						// Ignore parse errors for individual events
					}
				}
			}

			// Stream ended normally
			this.setState("disconnected");
			this.onDisconnected?.();

			// Attempt reconnect if not intentionally stopped
			if (!this.abortController.signal.aborted) {
				await this.handleConnectionDrop();
			}
		} catch (error) {
			if ((error as Error).name === "AbortError") {
				return;
			}
			this.setState("disconnected");
			this.onDisconnected?.();
			await this.handleConnectionDrop();
		}
	}

	/**
	 * Handle an unexpected connection drop.
	 * Attempts auto-restart if crash count is below the limit.
	 */
	private async handleConnectionDrop(): Promise<void> {
		if (!this.started) return;

		this.crashCount++;
		if (this.crashCount <= MAX_CRASH_RESTARTS) {
			console.debug("[TextStreamClient] Connection lost, attempting restart...");
			const ok = await this.restart();
			if (ok && this.loadMode === "auto") {
				this.connectSSE().catch(() => {});
			}
		} else {
			this.handleError(new Error("TextStream crashed multiple times. Disabling STT."));
			this.setState("error");
		}
	}

	/**
	 * Disconnect the SSE connection.
	 */
	private disconnectSSE(): void {
		if (this.abortController) {
			this.abortController.abort();
			this.abortController = null;
		}
	}

	// ------------------------------------------------------------------
	// Process management
	// ------------------------------------------------------------------

	/**
	 * Spawn the textstream Python process.
	 */
	private spawnProcess(): Promise<void> {
		return new Promise<void>((resolve, reject) => {
			const textstreamPath = path.join(this.pythonDir, "textstream");
			const proc = spawn(textstreamPath, ["--no-browser", "--port", String(this.port)], {
				stdio: ["ignore", "pipe", "pipe"],
			});

			this.process = proc;

			// Capture stdout for debug logging
			const stdoutChunks: Buffer[] = [];
			const stderrChunks: Buffer[] = [];
			proc.stdout?.on("data", (chunk: Buffer) => stdoutChunks.push(chunk));
			proc.stderr?.on("data", (chunk: Buffer) => stderrChunks.push(chunk));

			proc.on("error", (err) => {
				this.process = null;
				reject(err);
			});

			proc.on("exit", (code, _signal) => {
				this.process = null;
				if (code !== 0 && code !== null) {
					const stderr = Buffer.concat(stderrChunks).toString("utf-8");
					console.debug(`[TextStreamClient] Process exited with code ${code}: ${stderr}`);
				}
				// If the process dies unexpectedly while we're supposed to be running
				if (this.started && code !== 0) {
					this.handleConnectionDrop().catch(() => {});
				}
			});

			// Resolve immediately after spawn (health check happens separately)
			resolve();
		});
	}

	// ------------------------------------------------------------------
	// Health check
	// ------------------------------------------------------------------

	/**
	 * Check if the TextStream server is healthy.
	 * Returns true if GET /engine responds with 200.
	 */
	private async healthCheck(): Promise<boolean> {
		try {
			const res = await fetch(`http://127.0.0.1:${this.port}/engine`, {
				method: "GET",
				signal: AbortSignal.timeout(2000),
			});
			return res.ok;
		} catch {
			return false;
		}
	}

	/**
	 * Poll health check until the server responds or timeout.
	 */
	private async waitForHealth(): Promise<boolean> {
		const deadline = Date.now() + HEALTH_CHECK_TIMEOUT_MS;
		while (Date.now() < deadline) {
			if (await this.healthCheck()) {
				return true;
			}
			await new Promise((r) => setTimeout(r, HEALTH_CHECK_POLL_INTERVAL_MS));
		}
		return false;
	}

	// ------------------------------------------------------------------
	// State management
	// ------------------------------------------------------------------

	private setState(state: TextStreamClientState): void {
		this._state = state;
		this.onStateChange?.(state);
	}

	private handleError(error: Error): void {
		console.debug(`[TextStreamClient] Error: ${error.message}`);
		this.onError?.(error);
	}
}
