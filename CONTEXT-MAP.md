# Pi-Speech Context Map

This project is a monorepo (pi-mono) forked as **Pi-Speech** with integrated speech-to-text and text-to-speech. Each major package below has its own `CONTEXT.md` covering the domain language, architecture, and conventions specific to that area.

## Contexts

| Context | Location | Description |
|---------|----------|-------------|
| **Root** | `CONTEXT.md` | High-level project overview, cross-cutting domain glossary (STT, TTS, architecture), and shared conventions |
| **coding-agent** | `packages/coding-agent/CONTEXT.md` | The coding agent package — STT/TTS pipelines, interactive mode, slash commands, keybindings, services, settings, Python TTS worker |
| **ai** | `packages/ai/CONTEXT.md` | AI/LLM layer — model definitions, provider implementations, API registry, streaming |
| **agent** | `packages/agent/CONTEXT.md` | Core agent loop — message handling, tool execution, agent harness |
| **tui** | `packages/tui/CONTEXT.md` | Terminal UI framework — widgets, editor, keybindings, rendering |

## Reading order

1. Start with the root `CONTEXT.md` for a top-level understanding.
2. When working in a specific package, read that package's `CONTEXT.md` for local terminology and architecture.
3. For cross-package changes, read all relevant context files.

## ADRs

Architecture Decision Records live at `docs/adr/` (system-wide) and per-package `docs/adr/` directories where applicable.
