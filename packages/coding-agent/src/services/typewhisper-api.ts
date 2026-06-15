/**
 * HTTP client for the TypeWhisper macOS app REST API.
 *
 * TypeWhisper exposes an HTTP API on a dynamic port (auto-discovered via
 * a JSON discovery file written to ~/Library/Application Support/).
 *
 * Endpoints:
 *   POST /v1/dictation/start        – Begin push-to-talk dictation
 *   POST /v1/dictation/stop         – End dictation and transcribe
 *   GET  /v1/dictation/transcription?id=<uuid> – Poll for transcribed text
 *   GET  /v1/status                 – Check if engine is ready
 *   GET  /v1/models                 – List available STT engines/models
 *
 * Auto-launch: if the API is unreachable, launches TypeWhisper.app and polls
 * until ready (matching the behavior in whisper-bot's Python code).
 */

import { spawn } from "node:child_process";
import * as fs from "node:fs";
import * as os from "node:os";
import * as path from "node:path";

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

const DISCOVERY_PATHS = [
	// Dev build (built from local checkout via Xcode)
	path.join(os.homedir(), "Library/Application Support/TypeWhisper-Dev/api-discovery.json"),
	// Production build (installed .app)
	path.join(os.homedir(), "Library/Application Support/TypeWhisper/api-discovery.json"),
];

const DEFAULT_PORT = 8978;

interface DiscoveryData {
	port: number;
	token?: string;
}

let _discoveredPort: number | undefined;
let _discoveredToken: string | undefined;

function discover(): { port: number; token?: string } {
	if (_discoveredPort !== undefined) {
		return { port: _discoveredPort, token: _discoveredToken };
	}

	for (const discoveryPath of DISCOVERY_PATHS) {
		try {
			if (fs.existsSync(discoveryPath)) {
				const raw = fs.readFileSync(discoveryPath, "utf-8");
				const data = JSON.parse(raw) as DiscoveryData;
				_discoveredPort = data.port;
				_discoveredToken = data.token;
				return { port: data.port, token: data.token };
			}
		} catch {}
	}

	_discoveredPort = DEFAULT_PORT;
	return { port: DEFAULT_PORT };
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

export type TypeWhisperEngine = "parakeet" | "whisper" | string;

export interface TypeWhisperConfig {
	engine?: TypeWhisperEngine;
	port?: number;
	apiToken?: string;
}

export interface DictationResult {
	text: string;
	sessionId: string;
}

export interface StatusResult {
	ready: boolean;
	engine?: string;
	model?: string;
}

export interface ModelInfo {
	id: string;
	name: string;
	type: "stt";
	engine: string;
}

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

export class TypeWhisperError extends Error {
	public readonly statusCode?: number;

	constructor(message: string, statusCode?: number) {
		super(message);
		this.name = "TypeWhisperError";
		this.statusCode = statusCode;
	}
}

/**
 * HTTP client for TypeWhisper's REST API.
 */
export class TypeWhisperAPI {
	private baseUrl: string;
	private headers: Record<string, string>;

	constructor(config: TypeWhisperConfig = {}) {
		const { engine, port, apiToken } = config;
		const discovered = port ? { port, token: apiToken } : discover();

		this.baseUrl = `http://127.0.0.1:${discovered.port}`;
		this.headers = {
			"Content-Type": "application/json",
			"x-engine": engine ?? "parakeet",
		};
		if (discovered.token) {
			this.headers.Authorization = `Bearer ${discovered.token}`;
		}
	}

	// ------------------------------------------------------------------
	// Health / auto-launch (mirrors Python whisper-bot's _ensure_running)
	// ------------------------------------------------------------------

	/**
	 * Ensure TypeWhisper is running. Checks the API, and if unreachable,
	 * launches TypeWhisper.app and polls for up to 30 seconds until ready.
	 */
	async ensureRunning(): Promise<boolean> {
		// 1. Quick check — is the API already reachable?
		try {
			const res = await fetch(`${this.baseUrl}/v1/status`, {
				method: "GET",
				headers: this.headers,
				signal: AbortSignal.timeout(2000),
			});
			if (res.ok) return true;
		} catch {
			// Not reachable — fall through to launch
		}

		// 2. Not running — try to launch
		try {
			await new Promise<void>((resolve, reject) => {
				const proc = spawn("open", ["/Applications/TypeWhisper.app"], {
					stdio: "ignore",
				});
				proc.on("error", reject);
				proc.on("close", () => resolve());
			});
		} catch {
			return false;
		}

		// 3. Poll for up to 30 s (every 500 ms)
		const deadline = Date.now() + 30_000;
		while (Date.now() < deadline) {
			try {
				const res = await fetch(`${this.baseUrl}/v1/status`, {
					method: "GET",
					headers: this.headers,
					signal: AbortSignal.timeout(2000),
				});
				if (res.ok) return true;
			} catch {
				// Not ready yet
			}
			await new Promise((r) => setTimeout(r, 500));
		}

		return false;
	}

	/** Check if TypeWhisper is running and ready. */
	async status(): Promise<StatusResult> {
		try {
			const res = await fetch(`${this.baseUrl}/v1/status`, {
				method: "GET",
				headers: this.headers,
				signal: AbortSignal.timeout(2000),
			});
			if (!res.ok) return { ready: false };
			const data = (await res.json()) as Record<string, unknown>;
			return {
				ready: true,
				engine: data.engine as string | undefined,
				model: data.model as string | undefined,
			};
		} catch {
			return { ready: false };
		}
	}

	// ------------------------------------------------------------------
	// Dictation
	// ------------------------------------------------------------------

	/**
	 * Start a dictation session (push-to-talk). Returns the session UUID.
	 *
	 * If TypeWhisper responds with 409 "Already recording", we treat that
	 * as a success and still return the existing session ID if available.
	 */
	async startDictation(): Promise<string> {
		const res = await fetch(`${this.baseUrl}/v1/dictation/start`, {
			method: "POST",
			headers: this.headers,
		});

		if (res.status === 409) {
			// Already recording — extract the existing session ID from the response.
			// TypeWhisper may return it at body.id, body.session.id, or body.sessionId.
			const body = (await res.json()) as Record<string, unknown>;
			const sessionId = body.id || (body.session as Record<string, unknown> | undefined)?.id || body.sessionId;
			if (sessionId && typeof sessionId === "string") {
				return sessionId;
			}
			throw new TypeWhisperError(
				"Already recording but could not determine session ID. Stop the current session first.",
				409,
			);
		}

		if (!res.ok) {
			const text = await res.text();
			throw new TypeWhisperError(`TypeWhisper /dictation/start failed: ${res.status} ${text}`, res.status);
		}

		const data = (await res.json()) as { id: string };
		return data.id;
	}

	/** Stop dictation. Returns immediately; transcription is available via getTranscription(). */
	async stopDictation(sessionId: string): Promise<void> {
		const res = await fetch(`${this.baseUrl}/v1/dictation/stop`, {
			method: "POST",
			headers: this.headers,
			body: JSON.stringify({ id: sessionId }),
		});
		if (!res.ok) {
			throw new TypeWhisperError(
				`TypeWhisper /dictation/stop failed: ${res.status} ${await res.text()}`,
				res.status,
			);
		}
	}

	/**
	 * Poll for transcription result. Returns the transcribed text or empty string
	 * if not yet ready.
	 */
	async getTranscription(sessionId: string): Promise<string> {
		const url = `${this.baseUrl}/v1/dictation/transcription?id=${encodeURIComponent(sessionId)}`;
		const res = await fetch(url, {
			method: "GET",
			headers: this.headers,
		});
		if (!res.ok) {
			throw new TypeWhisperError(
				`TypeWhisper /dictation/transcription failed: ${res.status} ${await res.text()}`,
				res.status,
			);
		}
		const data = (await res.json()) as Record<string, unknown>;

		// TypeWhisper may return the text at different paths depending on version
		const text = (data.text as string) ?? ((data.transcription as Record<string, unknown>)?.text as string) ?? "";
		return text;
	}

	/**
	 * Poll for transcription with timeout. Returns transcribed text or throws.
	 */
	async waitForTranscription(sessionId: string, timeoutMs = 60_000): Promise<string> {
		const deadline = Date.now() + timeoutMs;
		while (Date.now() < deadline) {
			const text = await this.getTranscription(sessionId);
			if (text) return text;
			await new Promise((r) => setTimeout(r, 500));
		}
		throw new TypeWhisperError("TypeWhisper transcription timed out");
	}

	// ------------------------------------------------------------------
	// Models
	// ------------------------------------------------------------------

	/** List available STT models/engines. */
	async listModels(): Promise<ModelInfo[]> {
		const res = await fetch(`${this.baseUrl}/v1/models`, {
			method: "GET",
			headers: this.headers,
		});
		if (!res.ok) {
			throw new TypeWhisperError(`TypeWhisper /models failed: ${res.status} ${await res.text()}`, res.status);
		}
		const data = (await res.json()) as ModelInfo[];
		return data;
	}

	/** Force rediscovery (useful if TypeWhisper was restarted). */
	static rediscover(): void {
		_discoveredPort = undefined;
		_discoveredToken = undefined;
	}
}
