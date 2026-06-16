# coding-agent Domain Glossary

The coding agent package — the main entry point for the pi agent with integrated STT/TTS. This is where all speech pipeline work happens.

## Speech-to-Text (STT)

- **STTService** (`src/services/stt.ts`): Orchestrates dictation lifecycle — start, stop, interim transcription callbacks, parser integration. Backend-agnostic interface wrapping TypeWhisperAPI.
- **TypeWhisperAPI** (`src/services/typewhisper-api.ts`): HTTP client for the TypeWhisper macOS app. Auto-launches the app, discovers API port via JSON discovery file. Supports start/stop recording and text retrieval.
- **RegexT5Parser** (`src/services/regex-t5-parser.ts`): Two-stage STT output cleanup: (1) deterministic regex (filler removal, self-correction resolution, homophone fixes, punctuation normalization), (2) optional T5 ONNX grammar correction via HuggingFace transformers.
- **T5Service** (`src/services/t5-service.ts`): Lazy-loads Google's T5 ONNX model (`rabden/t5-tiny-gec-hone`, ~11MB quantized) for grammar correction. ~30-115ms CPU inference. Graceful degradation on error.
- **TextStream** (upcoming): Local streaming ASR server using Qwen3-ASR via MLX + Silero VAD. Intended to replace TypeWhisper.

## Text-to-Speech (TTS)

- **TTSService** (`src/services/tts.ts`): Spawns persistent Python subprocess (`whisper_bot/tts_worker.py`). Communicates via JSONL on stdin/stdout. Uses streaming providers with a two-phase protocol internally — `speak()` is fire-and-forget; `flush()` returns a promise that resolves when all queued sentences finish.
- **RegexTTSPolisher** (`src/services/tts-polisher.ts`): Deterministic text cleaner — strips thinking tags, code fences, markdown, list markers, humanizes identifiers (snake_case, kebab-case, camelCase, file paths), normalizes punctuation. Runs on each sentence before passing to the TTS worker. No LLM calls.

## Slash Commands

- `/stt` — on, off, auto, parser subcommands
- `/tts` — on, off, polish subcommands
- `/speed` — playback speed 0.5-3.0 step 0.25
- Defined in `src/modes/interactive/slash-commands.ts`

## Keybindings

- `app.recording.toggle` (Ctrl+Space) — start/stop STT dictation
- `app.tts.toggle` (none) — toggle TTS on/off
- Defined in `src/modes/interactive/keybindings.ts`

## Settings

Managed in `src/modes/interactive/settings-manager.ts`. Key flags: `sttEnabled`, `sttAutoSubmit`, `sttParserEnabled`, `ttsEnabled`, `ttsSpeed`, `ttsPolisherEnabled`.

## Services Index

Exported from `src/services/index.ts`: STTService, TTSService, RegexTTSPolisher, TypeWhisperAPI, RegexT5Parser.

## Python Worker

- `whisper_bot/tts_worker.py` — persistent worker (JSONL stdin/stdout)
- `whisper_bot/tts/piper.py` — Piper TTS with speed control and multi-voice
- `whisper_bot/tts/kokoro.py`, `tts/streaming_kokoro.py` — Kokoro TTS
- `whisper_bot/tts/sentence_splitter.py` — smart sentence segmentation
- `whisper_bot/parser/ollama.py` — Python Ollama parser (separate from the TS RegexT5Parser; calls Ollama HTTP API for qwen2.5-coder:3b). Used by the TTS pipeline, NOT the TUI mode.
