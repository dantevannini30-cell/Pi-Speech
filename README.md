<p align="center">
  <a href="https://pi.dev">
    <img alt="pi logo" src="https://pi.dev/logo-auto.svg" width="128">
  </a>
</p>
<p align="center">
  <a href="https://discord.com/invite/3cU7Bz4UPx"><img alt="Discord" src="https://img.shields.io/badge/discord-community-5865F2?style=flat-square&logo=discord&logoColor=white" /></a>
</p>
<p align="center">
  <a href="https://pi.dev">pi.dev</a> domain graciously donated by
  <br /><br />
  <a href="https://exe.dev"><img src="packages/coding-agent/docs/images/exy.png" alt="Exy mascot" width="48" /><br />exe.dev</a>
</p>

> New issues and PRs from new contributors are auto-closed by default. Maintainers review auto-closed issues daily. See [CONTRIBUTING.md](CONTRIBUTING.md).

---

# Pi-Speech

Fork of [pi-mono](https://github.com/earendil-works/pi-mono) with integrated speech-to-text (STT) and text-to-speech (TTS) for voice-driven coding sessions.

Speak your prompts, hear agent responses read aloud, and keep your hands on the keyboard for what matters.

## Key additions

### Speech-to-text (STT) via TypeWhisper

Dictate prompts instead of typing. Press **Ctrl+Space** to start recording, press again to stop, and the transcription is automatically submitted as a prompt.

- Uses the [TypeWhisper](https://typewhisper.com) macOS app REST API (auto-launches if not running)
- Supports configurable STT engines (Parakeet, Whisper)
- Runs raw transcription through a multi-stage parser:
  1. **Regex pass (always)**: removes filler words (um, uh, you know), resolves self-corrections ("wait no", "scratch that"), fixes homophones (their/they're, its/it's, to/too), normalizes punctuation and sentence casing
  2. **T5 grammar correction (optional)**: Google's T5 architecture via [Transformers.js](https://huggingface.co/rabden/t5-tiny-gec-hone) (~11MB quantized ONNX, ~30-115ms on CPU). Runs locally, no external API calls. Gracefully degrades if unavailable.
- Settings: `sttEnabled`, `sttAutoSubmit`, `sttParserEnabled`

### Text-to-speech (TTS) via Piper / Kokoro

Agent responses are spoken aloud as they stream in. A persistent Python worker keeps the TTS model loaded so there is no delay between sentences.

- **Piper TTS** (default) — fast, local, multi-voice. Runs as a persistent Python worker with speed control
- **Kokoro** — alternative TTS backend with streaming support (see `whisper_bot/tts/kokoro.py`, `tts/streaming_kokoro.py`)
- **macOS `say` fallback** when Python/Piper is unavailable
- Adjust playback speed with `/speed 0.5` to `/speed 3.0` (step 0.25)
- Settings: `ttsEnabled`, `ttsSpeed`

### TTS polisher

Before speaking, agent output is cleaned up so it sounds natural — no "```python" or "**bold**" read aloud.

- Strips thinking/scratchpad tags, code fences, markdown formatting, inline code, file paths, and list markers
- Purely regex-based (no LLM calls) — fast, deterministic, zero network calls
- Humanizes identifiers: `snake_case`, `kebab-case`, `camelCase` and file paths are converted to natural speech
- Settings: `ttsPolisherEnabled`

### Slash commands

| Command | Description |
|---------|-------------|
| `/stt` | STT commands: `on`, `off`, `auto` (toggle auto-submit) |
| `/tts` | TTS commands: `on`, `off` |
| `/speed` | Set TTS playback speed: `0.5` – `3.0` (step 0.25) |

### Keybindings

| Shortcut | Action |
|----------|--------|
| **Ctrl+Space** | Toggle STT recording (start/stop dictation) |

TTS toggle has no default keybinding — assign one via `app.tts.toggle` in your keybindings config.

### Voice settings

All voice settings are accessible via the settings UI (`/settings`):

| Setting | Default | Description |
|---------|---------|-------------|
| `sttEnabled` | `true` | Enable speech-to-text (requires TypeWhisper) |
| `ttsEnabled` | `true` | Enable text-to-speech |
| `sttAutoSubmit` | `true` | Auto-submit transcription when recording stops |
| `sttParserEnabled` | `true` | Clean STT output (regex + T5 grammar correction) |
| `ttsSpeed` | `1.0` | TTS playback speed (0.5-3.0, step 0.25) |
| `ttsPolisherEnabled` | `true` | Polish TTS output before speaking (regex-based) |

### Architecture

```
┌─────────────────────────────────────────────────────┐
│  Pi agent (TypeScript)                               │
│                                                       │
│  STTService ──▶ TypeWhisperAPI ──▶ TypeWhisper macOS  │
│       │                                                │
│       ▼                                                │
│  RegexT5Parser                                          │
│     ├─ 1. Regex pass (fillers, corrections, homophones)│
│     └─ 2. T5 grammar correction (rabden/t5-tiny-gec,   │
│             Transformers.js/ONNX, no API calls)          │
│                                                         │
│  TTSService ──▶ Python worker (Piper/Kokoro) ──▶ Audio  │
│       │                                                  │
│       ▼                                                  │
│  RegexTTSPolisher (no LLM, no network)                   │
└─────────────────────────────────────────────────────┘
```

## All packages

| Package | Description |
|---------|-------------|
| **[@earendil-works/pi-ai](packages/ai)** | Unified multi-provider LLM API (OpenAI, Anthropic, Google, etc.) |
| **[@earendil-works/pi-agent-core](packages/agent)** | Agent runtime with tool calling and state management |
| **[@earendil-works/pi-coding-agent](packages/coding-agent)** | Interactive coding agent CLI (with STT/TTS integration) |
| **[@earendil-works/pi-tui](packages/tui)** | Terminal UI library with differential rendering |
| **whisper_bot** | Python TTS workers (Piper, Kokoro) and supporting modules |

The Python TTS workers live in `packages/coding-agent/whisper_bot/tts/`:
- `piper.py` — Piper TTS with speed control and multi-voice support
- `kokoro.py` — Kokoro TTS backend
- `streaming_kokoro.py` — Streaming Kokoro for low-latency playback
- `sentence_splitter.py` — Smart sentence segmentation for TTS
- `tts_worker.py` — Persistent worker process (JSONL protocol on stdin/stdout)
- `tts_cli.py` — Standalone TTS CLI
- `tts_streaming_cli.py` — Standalone streaming TTS CLI

## Prerequisites

- **STT**: [TypeWhisper](https://typewhisper.com) macOS app
- **TTS**: Python 3, [Piper TTS](https://github.com/rhasspy/piper) or Kokoro, or macOS `say`
- **STT parser**: No external services needed — the T5 grammar correction model (~11MB) is downloaded on first use and cached via HuggingFace
- **TTS polisher**: Zero dependencies — purely regex-based

## Quick start

```bash
npm install --ignore-scripts
npm run build
./pi-test.sh              # Run from source
```

Once running, press **Ctrl+Space** to start dictating, or enable TTS with `/tts on`.

## Permissions & Containerization

Pi does not include a built-in permission system for restricting filesystem, process, network, or credential access. By default, it runs with the permissions of the user and process that launched it.

If you need stronger boundaries, containerize or sandbox Pi. See [packages/coding-agent/docs/containerization.md](packages/coding-agent/docs/containerization.md) for three patterns:

- **Gondolin extension**: keep `pi` and provider auth on the host while routing built-in tools and `!` commands into a local Linux micro-VM.
- **Plain Docker**: run the whole `pi` process in a local container for simple isolation.
- **OpenShell**: run the whole `pi` process in a policy-controlled sandbox.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines and [AGENTS.md](AGENTS.md) for project-specific rules (for both humans and agents).

## Development

```bash
npm install --ignore-scripts  # Install all dependencies without running lifecycle scripts
npm run build        # Build all packages
npm run check        # Lint, format, and type check
./test.sh            # Run tests (skips LLM-dependent tests without API keys)
./pi-test.sh         # Run pi from sources (can be run from any directory)
```

## Supply-chain hardening

We treat npm dependency changes as reviewed code changes.

- Direct external dependencies are pinned to exact versions. Internal workspace packages remain version-ranged.
- `.npmrc` sets `save-exact=true` and `min-release-age=2` to avoid same-day dependency releases during npm resolution.
- `package-lock.json` is the dependency ground truth. Pre-commit blocks accidental lockfile commits unless `PI_ALLOW_LOCKFILE_CHANGE=1` is set.
- `npm run check` verifies pinned direct deps, native TypeScript import compatibility, and the generated coding-agent shrinkwrap.
- The published CLI package includes `packages/coding-agent/npm-shrinkwrap.json`, generated from the root lockfile, to pin transitive deps for npm users.
- Release smoke tests use `npm run release:local` to build, pack, and create isolated npm and Bun installs outside the repo before tagging a release.
- Local release installs, documented npm installs, and `pi update --self` use `--ignore-scripts` where supported.
- CI installs with `npm ci --ignore-scripts`, and a scheduled GitHub workflow runs `npm audit --omit=dev` plus `npm audit signatures --omit=dev`.
- Shrinkwrap generation has an explicit allowlist for dependency lifecycle scripts; new lifecycle-script deps fail checks until reviewed.

## License

MIT
