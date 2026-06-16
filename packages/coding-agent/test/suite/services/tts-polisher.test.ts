/**
 * Tests for the TTS polisher (RegexTTSPolisher).
 *
 * Pure regex-based polisher — no LLM, no network.
 * Note: the polisher skips text under 30 characters, so all meaningful
 * test inputs must be >= 30 chars.
 */

import { describe, expect, it } from "vitest";
import {
	DEFAULT_TTS_POLISHER_CONFIG,
	RegexTTSPolisher,
	type TTSPolisherConfig,
} from "../../../src/services/tts-polisher.ts";

function createPolisher(overrides: Partial<TTSPolisherConfig> = {}): RegexTTSPolisher {
	return new RegexTTSPolisher({ ...DEFAULT_TTS_POLISHER_CONFIG, ...overrides });
}

describe("RegexTTSPolisher", () => {
	describe("constructor", () => {
		it("stores config", () => {
			const polisher = createPolisher({ enabled: false });
			expect(polisher.config.enabled).toBe(false);
		});

		it("uses defaults from DEFAULT_TTS_POLISHER_CONFIG", () => {
			const polisher = createPolisher();
			expect(polisher.config.enabled).toBe(true);
		});
	});

	describe("polish", () => {
		it("strips thinking tags and their content", async () => {
			const polisher = createPolisher();
			const text = "Hello <thinking>this is a thought process</thinking> world, how are you today?";
			const result = await polisher.polish(text);
			expect(result).toContain("Hello");
			expect(result).toContain("world");
			expect(result).not.toContain("<thinking>");
			expect(result).not.toContain("</thinking>");
		});

		it("strips <think> tags (short form)", async () => {
			const polisher = createPolisher();
			const text = "Hello <think>quick thought</think> world, how are you doing today?";
			const result = await polisher.polish(text);
			expect(result).toContain("Hello");
			expect(result).not.toContain("<think>");
			expect(result).not.toContain("</think>");
		});

		it("strips fenced code blocks", async () => {
			const polisher = createPolisher();
			const text = "Here is some text with a code block:\n```\nconst x = 1;\n```\ncontinuing here.";
			const result = await polisher.polish(text);
			expect(result).toContain("code snippet");
			expect(result).not.toContain("```");
		});

		it("removes markdown link syntax keeping link text", async () => {
			const polisher = createPolisher();
			const text = "Please check the documentation at [docs](https://example.com) for more details.";
			const result = await polisher.polish(text);
			expect(result).toContain("docs");
			expect(result).not.toContain("https://example.com");
			expect(result).not.toContain("[docs]");
		});

		it("removes inline code backticks keeping the content", async () => {
			const polisher = createPolisher();
			const text = "To install the package you should run `npm install @my/package` in your terminal window.";
			const result = await polisher.polish(text);
			expect(result).toContain("npm install");
			expect(result).not.toContain("`npm install");
		});

		it("removes markdown heading markers", async () => {
			const polisher = createPolisher();
			const text = "# Main Title\n## Section Title\nHere is some body text that explains the section.";
			const result = await polisher.polish(text);
			expect(result).toContain("Main Title");
			expect(result).toContain("Section Title");
			expect(result).toContain("body text");
			expect(result).not.toMatch(/^#/m);
		});

		it("removes bold and italic markers", async () => {
			const polisher = createPolisher();
			const text = "This text has **bold words** and *italic phrases* throughout the content.";
			const result = await polisher.polish(text);
			expect(result).toContain("bold words");
			expect(result).toContain("italic phrases");
			expect(result).not.toContain("**");
			expect(result).not.toContain("*italic");
		});

		it("strips unordered list markers", async () => {
			const polisher = createPolisher();
			const text = "The items are:\n- first item in the list\n- second item in the list\n- third item here";
			const result = await polisher.polish(text);
			expect(result).toContain("first item in the list");
			expect(result).toContain("second item in the list");
			expect(result).toContain("third item here");
			expect(result).not.toMatch(/^- /m);
		});

		it("strips ordered list markers", async () => {
			const polisher = createPolisher();
			const text = "Steps to follow:\n1. first step in process\n2. second step in process\n3. third step in process";
			const result = await polisher.polish(text);
			expect(result).toContain("first step in process");
			expect(result).toContain("second step in process");
			expect(result).toContain("third step in process");
			expect(result).not.toMatch(/^\d+\. /m);
		});

		it("strips diff +/- markers", async () => {
			const polisher = createPolisher();
			const text = "Changes made:\n+ added this new line of code\n- removed this old line\nnormal line here";
			const result = await polisher.polish(text);
			expect(result).toContain("added this new line of code");
			expect(result).toContain("removed this old line");
			expect(result).toContain("normal line here");
			expect(result).not.toMatch(/^[+-] /m);
		});

		it("humanizes snake_case identifiers", async () => {
			const polisher = createPolisher();
			const text = "Please open the file called my_file_name_and_open_it in the editor for review.";
			const result = await polisher.polish(text);
			expect(result).toContain("my file name and open it");
		});

		it("humanizes kebab-case identifiers", async () => {
			const polisher = createPolisher();
			const text = "The configuration file is called my-config-file and lives in the root directory here.";
			const result = await polisher.polish(text);
			expect(result).toContain("my config file");
		});

		it("humanizes camelCase identifiers", async () => {
			const polisher = createPolisher();
			const text = "You should call the myFunctionName function to start the processing pipeline here.";
			const result = await polisher.polish(text);
			expect(result).toContain("my Function Name");
		});

		it("replaces forward slashes with 'slash'", async () => {
			const polisher = createPolisher();
			const text = "The file is located at /home/user/documents and you should open it from there.";
			const result = await polisher.polish(text);
			expect(result).toContain("slash home slash user slash documents");
		});

		it("replaces dots between words with 'dot'", async () => {
			const polisher = createPolisher();
			const text = "Please read the file called readme.txt for setup instructions and guidance.";
			const result = await polisher.polish(text);
			expect(result).toContain("readme dot txt");
		});

		it("normalizes repeated exclamation/question marks", async () => {
			const polisher = createPolisher();
			const text = "The user asked Really!!! What do you mean?? I am not sure about this question.";
			const result = await polisher.polish(text);
			expect(result).toContain("!");
			expect(result).toContain("?");
			expect(result).not.toContain("!!!");
			expect(result).not.toContain("??");
		});

		it("strips HTML entities", async () => {
			const polisher = createPolisher();
			const text = "The company AT&amp;T uses &lt;bold&gt; tags in their documentation files here.";
			const result = await polisher.polish(text);
			expect(result).toContain("AT&T");
			expect(result).toContain("<bold>");
		});

		it("collapses multiple spaces and trims whitespace", async () => {
			const polisher = createPolisher();
			const text = "  This   text    has  lots   of   extra     spaces.    ";
			const result = await polisher.polish(text);
			expect(result).toBe("This text has lots of extra spaces.");
		});

		it("removes blockquote markers", async () => {
			const polisher = createPolisher();
			const text = "The speaker said:\n> quoted text from the document\nand then continued speaking.";
			const result = await polisher.polish(text);
			expect(result).toContain("quoted text from the document");
			expect(result).not.toMatch(/^>/m);
		});

		it("returns short text under 30 chars unchanged", async () => {
			const polisher = createPolisher();
			const text = "Sure!";
			const result = await polisher.polish(text);
			expect(result).toBe("Sure!");
		});

		it("polishes text at exactly 30 chars", async () => {
			const polisher = createPolisher();
			const text = "a".repeat(30);
			const result = await polisher.polish(text);
			expect(result.length).toBeGreaterThan(0);
		});
	});
});
