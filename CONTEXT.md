# Pi-Speech Domain Glossary

## Speech-to-Text (STT)

- **TypeWhisper**: macOS dictation app with HTTP REST API. Current STT backend. Does not support streaming/interim transcription. Targeted for replacement.
- **TextStream**: Local streaming ASR server (`textstream-asr` pip package). Runs Qwen3-ASR (0.6B default) via MLX on Apple Silicon, with Silero VAD. Exposes SSE endpoint at `localhost:7890/stream`. Intended replacement for TypeWhisper.
- **Push-to-talk (PTT)**: Interaction model where recording starts/stops on explicit user action (Ctrl+Space). TextStream is paused/resumed via `/pause` and `/resume` HTTP endpoints to implement PTT.
- **finalized**: Text in TextStream's SSE output that the model has confirmed and will not change. Displayed in the editor in real-time during recording.
- **draft**: Speculative text in TextStream's SSE output that the model is still refining. Not displayed during recording; merged onto finalized text when PTT ends (after 200ms drain timeout).
- **Auto-load**: STT process model option where TextStream is spawned as soon as STT is enabled.
- **Lazy-load**: STT process model option where TextStream is spawned on first Ctrl+Space press.
- **SSE client**: Persistent HTTP connection to TextStream's `/stream` endpoint. Each event carries `finalized` and `draft` fields. Used instead of polling for lower streaming latency.

## Text-to-Speech (TTS)

- **Piper**: Lightweight neural TTS engine. Runs via Python subprocess (`tts_worker.py`). Supports multiple voices and speed control.
- **Kokoro**: Higher-quality TTS engine (ONNX-based). Also runs via the same `tts_worker.py`.
- **RegexTTSPolisher**: Deterministic text polisher that strips thinking tags, code fences, markdown, and normalizes identifiers before TTS.
- **getVenvPython()**: Returns path to `.venv/bin/python3` inside `packages/coding-agent/`. Used for all Python subprocess spawning.

## Architecture

- **STTService** (`stt.ts`): Orchestrates dictation lifecycle — start, stop, interim transcription callbacks, parser integration. Backend-agnostic.
- **TextStreamClient** (new): HTTP+SSE client for TextStream. Manages process lifecycle, SSE connection, pause/resume, health checks.
- **TypeWhisperAPI** (`typewhisper-api.ts`): HTTP client for TypeWhisper. To be removed when migration is complete.
- **interactive-mode.ts**: Wires STTService callbacks to TUI editor. `onInterimTranscription` updates editor text; final text is submitted via Enter simulation.
- **.venv**: Python virtual environment at `packages/coding-agent/.venv/`. Shared between TTS worker and TextStream.
