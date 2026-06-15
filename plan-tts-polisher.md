# Plan: Ollama-backed TTS Output Polisher

## Problem

Agent output contains markdown formatting (`# headings`, `**bold**`, `` `code` ``, ```code blocks```, `[links]()`, etc.), thinking tags, and tool call metadata. When fed directly to TTS (piper/say), this text sounds unnatural — punctuation is read aloud, code is spoken verbatim, and formatting noise breaks the listening experience.

## Current State

- `TTSService` (in `services/tts.ts`) sends raw text to piper/say via a Python worker.
- In `InteractiveMode`, the `message_update` handler extracts text from assistant messages via `extractTextFromAssistantMessage()` and queues it in `ttsBuffer`. On sentence boundaries or `message_end`, `flushTtsBuffer()` sends it to `TTSService.speak()`.
- **No cleanup is applied** — text goes straight from agent to TTS.
- The other agent is adding an `OllamaParser` / `ParserProvider` for cleaning STT input. This plan mirrors that pattern for the output side.

## Proposed Design

### 1. New Service: `TTSPolisherProvider` interface + `OllamaTTSPolisher`

**Location**: `packages/coding-agent/src/services/tts-polisher.ts`

Interface (minimal, same shape as `ParserProvider`):

```ts
export interface TTSPolisherConfig {
  enabled: boolean;
  endpoint: string;       // Ollama-compatible API (e.g. http://localhost:11434/v1)
  model: string;          // e.g. qwen2.5:1.5b
  timeoutMs: number;
}

export interface TTSPolisherProvider {
  readonly config: TTSPolisherConfig;
  polish(text: string, signal?: AbortSignal): Promise<string>;
}
```

**`OllamaTTSPolisher`** class:

- Sends text to `{endpoint}/chat/completions` with a **system prompt** optimized for TTS-friendly output cleaning.
- Uses `temperature: 0`, `max_tokens` based on input length.
- Falls back to raw text on any error (graceful degradation — same pattern as `OllamaParser`).
- Optional `postProcess()` for simple regex fixes the LLM might miss.

**System Prompt** (draft):

```
You are a text-to-speech text cleaner. Your ONLY job is to rewrite text so it
sounds natural when read aloud. You are NOT a chatbot — do NOT respond, answer,
or explain.

Rules:
1. Remove all markdown formatting: headings (#), bold (**), italic (*),
   inline code (`), code blocks (```), blockquotes (>), horizontal rules (---),
   tables, and link syntax [text](url) — keep only the link text.
2. Remove AI thinking/scratchpad content (content between think tags or
   other internal AI reasoning).
3. Clean up code: when code is inline, read it as natural language; for code
   blocks, describe them briefly (e.g., "code block omitted" or summarize
   what the code does if it's short enough to be intelligible when spoken).
4. Normalize punctuation: replace em-dashes with "dash", remove decorative
   characters (===, ***, arrows like ->), ensure sentences end with periods.
5. Remove bullet markers (*, -) and numbered list markers — keep the content
   as flowing prose.
6. Remove file paths and diff syntax that are noisy when spoken. Replace
   paths with descriptive references where possible.
7. Keep all factual content, instructions, and meaning intact. Do not
   summarize, rephrase opinions, or change the message.

Output ONLY the cleaned text. No greetings, no explanations, no markdown.
```

### 2. Settings

Add TTS polisher settings to `SettingsManager` (stored in `settings.json`):

| Key | Type | Default | Description |
|---|---|---|---|
| `ttsPolisherEnabled` | `boolean` | `true` | Enable/disable TTS text polishing |
| `ttsPolisherEndpoint` | `string` | `http://localhost:11434/v1` | Ollama endpoint |
| `ttsPolisherModel` | `string` | `qwen2.5:1.5b` | Model name |
| `ttsPolisherTimeoutMs` | `number` | `3000` | Per-request timeout |

These could share defaults with the STT parser settings, or be independent. Independent gives more flexibility (different models for input vs output cleanup). Either way is fine.

### 3. Integration in `InteractiveMode`

**Location**: `packages/coding-agent/src/modes/interactive/interactive-mode.ts`

Current flow (simplified):

```
message_update
  -> extractTextFromAssistantMessage() -> raw text
  -> ttsBuffer += newText
  -> flushTtsBuffer() -> ttsService.speak(text)
```

New flow:

```
message_update
  -> extractTextFromAssistantMessage() -> raw text
  -> ttsBuffer += newText
  -> flushTtsBuffer()
       -> if ttsPolisher.enabled: ollamaTTSPolisher.polish(text) -> cleaned text
       -> ttsService.speak(cleanedText)
```

Key changes in `InteractiveMode`:

1. **Constructor**: instantiate `OllamaTTSPolisher` with settings, similar to how `OllamaParser` is created for STT.
2. **`flushTtsBuffer()`**: add a polisher step before `ttsService.speak()`. The polish call is async, so the existing `ttsSpeakChain` handles sequencing.
3. **Settings wiring**: on TTS toggle or settings change, update the polisher's config.
4. **Settings manager methods**: add getters/setters for the four new settings keys.

### 4. Lazy / Cached Calls

Polishing every sentence boundary is wasteful. Strategies:

- **Sentence-level**: only polish when a complete sentence boundary is reached (current approach already buffers to sentence boundaries). Each sentence gets polished independently.
- **Debounce**: if the user wants, add a short debounce to batch adjacent sentences. Not needed for v1 — sentence boundaries are fine.
- **Skip short fragments**: fragments under ~30 chars pass through unpolished (they're likely punctuation or short acknowledgements like "Sure!" or "Done.").

### 5. Error Handling

- Network timeout / model not loaded / any exception → return raw text (graceful degradation).
- Empty output from polisher → return raw text as fallback.
- Non-JSON endpoint responses → return raw text.
- All failures are silent (logged to debug if available). The user's listening experience is never blocked by a polisher failure.

### 6. Testing

New test file: `packages/coding-agent/test/suite/services/tts-polisher.test.ts`

- Unit test `OllamaTTSPolisher` with mock fetch.
- Test system prompt inclusion.
- Test graceful degradation on network error.
- Test post-processing regex fixes.

Integration test in interactive-mode flow: verify that when TTS is enabled and polisher is enabled, the text fed to `TTSService.speak()` is the polished version, not the raw version.

### 7. Implementation Order

1. Create `TTSPolisherProvider` interface and `OllamaTTSPolisher` class in `services/tts-polisher.ts`
2. Add settings keys to `SettingsManager` (getters/setters + defaults)
3. Wire `OllamaTTSPolisher` into `InteractiveMode` constructor and `flushTtsBuffer()`
4. Export from `services/index.ts`
5. Write tests
6. Run `npm run check`, fix any issues

### 8. Open Questions / Edge Cases

- **Should tool execution output be polished too?** Tool results often contain code, file diffs, and paths. If TTS reads them, polishing would help. But tool output inclusion in TTS is already a design choice — this plan focuses on the assistant's main text content only.
- **Model size vs latency**: `qwen2.5:1.5b` is fast (~100-300ms per sentence on modern hardware). If latency is a concern, smaller models like `tinyllama` or `phi3:mini` work. This is configurable via `ttsPolisherModel`.
- **Streaming vs per-sentence**: The current TTS buffers to sentence boundaries then sends the whole sentence. Polishing happens just before sending. No streaming from the polisher needed — send the sentence text, get cleaned text back, feed to TTS.
- **What about `/bye`, `/exit`, shortcut responses?** These are short and typically formatting-free. The "skip fragments under 30 chars" heuristic handles them.

## Files to Create / Modify

| Action | File |
|---|---|
| **Create** | `packages/coding-agent/src/services/tts-polisher.ts` |
| **Modify** | `packages/coding-agent/src/services/index.ts` (export new types/class) |
| **Modify** | `packages/coding-agent/src/core/settings-manager.ts` (new settings keys) |
| **Modify** | `packages/coding-agent/src/modes/interactive/interactive-mode.ts` (wire polisher into TTS flow) |
| **Create** | `packages/coding-agent/test/suite/services/tts-polisher.test.ts` (tests) |
