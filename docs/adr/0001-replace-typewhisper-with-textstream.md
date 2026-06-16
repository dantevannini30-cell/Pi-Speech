# Add streaming STT by replacing TypeWhisper with TextStream

TypeWhisper (the macOS dictation app used as the STT backend) only returns a single transcription chunk after recording stops — it has no support for interim/partial results. To get real-time streaming of transcribed text into the TUI editor while the user speaks, we replace it with TextStream, a local streaming ASR server that runs Qwen3-ASR via MLX on Apple Silicon and streams finalized text over Server-Sent Events.

The alternatives were:
- **Whisper.cpp with streaming mode**: cross-platform but slower on Apple Silicon than MLX-based solutions, and requires manual chunk management for SSE output.
- **Realtime_mlx_STT**: full-featured library with streaming and wake word support, but heavier — more deps and complexity than needed for a simple pause/resume dictation flow.
- **MLX Whisper (polling)**: would require us to build the buffering, VAD, and streaming logic ourselves. TextStream bundles all of that into a single `pip install`.
- **Staying with TypeWhisper**: simple and working, but fundamentally cannot stream interim results — the API only returns text after stop.

TextStream was chosen because it is the lightest path to a working streaming endpoint: single dependency, exposes SSE out of the box, and runs efficiently on Apple Silicon via MLX.

Status: accepted

Consequences:
- TextStream runs as a long-lived Python subprocess alongside the TTS worker in the same `.venv`, adding ~1.2GB RAM usage while active.
- The push-to-talk lifecycle changes from start/stop to pause/resume.
- We gain the ability to show live transcription in the editor as the user speaks.
- TypeWhisper is removed entirely (no fallback).
