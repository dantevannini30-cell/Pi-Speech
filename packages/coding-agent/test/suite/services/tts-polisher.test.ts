/**
 * Tests for the TTS polisher (OllamaTTSPolisher).
 *
 * These are unit tests that mock the fetch API to avoid needing
 * a real Ollama endpoint.
 */

import { afterEach, describe, expect, it, vi } from "vitest";
import {
	DEFAULT_TTS_POLISHER_CONFIG,
	OllamaTTSPolisher,
	type TTSPolisherConfig,
} from "../../../src/services/tts-polisher.ts";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function createMockFetch(responseBody: unknown, status = 200): typeof fetch {
	return vi.fn().mockResolvedValue({
		ok: status >= 200 && status < 300,
		status,
		json: vi.fn().mockResolvedValue(responseBody),
	});
}

function createPolisher(overrides: Partial<TTSPolisherConfig> = {}): OllamaTTSPolisher {
	return new OllamaTTSPolisher({ ...DEFAULT_TTS_POLISHER_CONFIG, ...overrides });
}

/** Shortcut for a mock fetch that returns a given cleaned string. */
function mockFetchReturns(cleaned: string): void {
	vi.stubGlobal(
		"fetch",
		createMockFetch({
			choices: [{ message: { content: cleaned } }],
		}),
	);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe("OllamaTTSPolisher", () => {
	afterEach(() => {
		vi.unstubAllGlobals();
	});

	describe("constructor", () => {
		it("stores config", () => {
			const polisher = createPolisher({ model: "tinyllama", timeoutMs: 5000 });
			expect(polisher.config.model).toBe("tinyllama");
			expect(polisher.config.timeoutMs).toBe(5000);
			expect(polisher.config.endpoint).toBe("http://localhost:11434/v1");
		});

		it("uses defaults from DEFAULT_TTS_POLISHER_CONFIG", () => {
			const polisher = createPolisher();
			expect(polisher.config.enabled).toBe(true);
			expect(polisher.config.endpoint).toBe("http://localhost:11434/v1");
			expect(polisher.config.model).toBe("qwen2.5:1.5b");
			expect(polisher.config.timeoutMs).toBe(3000);
		});
	});

	describe("polish", () => {
		it("returns cleaned text from the LLM", async () => {
			mockFetchReturns("This is cleaned text for TTS output.");
			const polisher = createPolisher();
			const result = await polisher.polish("This text has **bold** and `code` that needs cleaning.");
			expect(result).toBe("This is cleaned text for TTS output.");
		});

		it("sends the correct payload", async () => {
			const fetchMock = vi.fn().mockResolvedValue({
				ok: true,
				json: vi.fn().mockResolvedValue({
					choices: [{ message: { content: "cleaned" } }],
				}),
			});
			vi.stubGlobal("fetch", fetchMock);

			const polisher = createPolisher({ model: "test-model" });
			await polisher.polish("This is some raw text that needs cleaning for TTS output.");

			expect(fetchMock).toHaveBeenCalledTimes(1);
			const [url, options] = fetchMock.mock.calls[0] as [string, RequestInit];
			expect(url).toBe("http://localhost:11434/v1/chat/completions");
			const body = JSON.parse(options.body as string);
			expect(body.model).toBe("test-model");
			expect(body.temperature).toBe(0);
			expect(body.messages).toHaveLength(2);
			expect(body.messages[0].role).toBe("system");
			expect(body.messages[1].role).toBe("user");
			expect(body.messages[1].content).toBe("This is some raw text that needs cleaning for TTS output.");
		});

		it("strips trailing slashes from endpoint", async () => {
			const fetchMock = vi.fn().mockResolvedValue({
				ok: true,
				json: vi.fn().mockResolvedValue({
					choices: [{ message: { content: "cleaned" } }],
				}),
			});
			vi.stubGlobal("fetch", fetchMock);

			const polisher = createPolisher({ endpoint: "http://localhost:11434/v1/" });
			await polisher.polish("This is some text that needs cleaning for TTS output.");
			const [url] = fetchMock.mock.calls[0] as [string];
			expect(url).toBe("http://localhost:11434/v1/chat/completions");
		});

		it("returns raw text when response is not ok (graceful degradation)", async () => {
			vi.stubGlobal("fetch", createMockFetch({}, 500));
			const polisher = createPolisher();
			const result = await polisher.polish("This text has **bold** and `code` that needs cleaning.");
			expect(result).toBe("This text has **bold** and `code` that needs cleaning.");
		});

		it("returns raw text when response has no choices", async () => {
			vi.stubGlobal("fetch", createMockFetch({}));
			const polisher = createPolisher();
			const result = await polisher.polish("This text has **bold** and `code` that needs cleaning.");
			expect(result).toBe("This text has **bold** and `code` that needs cleaning.");
		});

		it("returns raw text on fetch error", async () => {
			vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("network error")));
			const polisher = createPolisher();
			const result = await polisher.polish("This text has **bold** and `code` that needs cleaning.");
			expect(result).toBe("This text has **bold** and `code` that needs cleaning.");
		});

		it("returns raw text on timeout", async () => {
			vi.stubGlobal(
				"fetch",
				vi.fn().mockRejectedValue(new DOMException("The operation was aborted", "TimeoutError")),
			);
			const polisher = createPolisher();
			const result = await polisher.polish("This text has **bold** and `code` that needs cleaning.");
			expect(result).toBe("This text has **bold** and `code` that needs cleaning.");
		});

		it("skips polishing for very short text under 30 chars", async () => {
			const fetchMock = vi.fn();
			vi.stubGlobal("fetch", fetchMock);
			const polisher = createPolisher();
			const result = await polisher.polish("Sure!");
			expect(result).toBe("Sure!");
			expect(fetchMock).not.toHaveBeenCalled();
		});

		it("polishes text at exactly 30 chars", async () => {
			const fetchMock = vi.fn().mockResolvedValue({
				ok: true,
				json: vi.fn().mockResolvedValue({
					choices: [{ message: { content: "polished" } }],
				}),
			});
			vi.stubGlobal("fetch", fetchMock);
			const polisher = createPolisher();
			const text = "a".repeat(30);
			await polisher.polish(text);
			expect(fetchMock).toHaveBeenCalled();
		});
	});

	describe("postProcess", () => {
		it("removes markdown heading markers", async () => {
			mockFetchReturns("# Title\n## Subtitle\nBody text content here.");
			const polisher = createPolisher();
			const result = await polisher.polish("Some raw text with markdown content for TTS output.");
			expect(result).not.toContain("##");
			expect(result).toContain("Title");
			expect(result).toContain("Subtitle");
			expect(result).toContain("Body text content here.");
		});

		it("removes backtick code markers keeping content", async () => {
			mockFetchReturns("To install run `npm install` in your terminal directory.");
			const polisher = createPolisher();
			const result = await polisher.polish("Some raw text with backtick content for TTS output.");
			expect(result).toContain("npm install");
			expect(result).not.toContain("`npm install`");
		});

		it("removes link syntax keeping link text", async () => {
			mockFetchReturns("See [docs](https://example.com) for more information on this.");
			const polisher = createPolisher();
			const result = await polisher.polish("Some raw text with link content for TTS output.");
			expect(result).toContain("See docs for more information on this.");
			expect(result).not.toContain("https://example.com");
		});
	});
});
