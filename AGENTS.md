# Development Rules

## Pi-Speech Architecture

This is **Pi-Speech**, a fork of pi-mono with integrated STT (speech-to-text) and TTS (text-to-speech).

### STT pipeline (microphone -> prompt)

```
TypeWhisper macOS app ──HTTP──▶ STTService ──▶ OllamaParser (regex + T5 ONNX) ──▶ agent prompt
```

- `STTService` (`packages/coding-agent/src/services/stt.ts`) wraps `TypeWhisperAPI` (`typewhisper-api.ts`) to talk to the TypeWhisper macOS app REST API
- `TypeWhisperAPI` auto-launches TypeWhisper.app if not running, discovers the API port via a JSON discovery file
- **`RegexT5Parser`** (`regex-t5-parser.ts`) uses:
  1. **Deterministic regex pass**: removes filler words (um, uh, you know), resolves self-corrections ("wait no", "scratch that"), fixes homophones (their/they're, its/it's, to/too), normalizes punctuation and sentence casing
  2. **Optional T5 grammar correction**: Google's T5 architecture via `@huggingface/transformers` (ONNX runtime). Model is `rabden/t5-tiny-gec-hone` (~11MB quantized, ~30-115ms CPU inference). Imported lazily on first use. Cached in `~/.cache/huggingface`. Graceful degradation on error.
- `t5-service.ts` handles the lazy model loading and `correctGrammar()` function
- `ParserConfig` still defines `endpoint`, `model`, `timeoutMs` fields (legacy from earlier Ollama-based design) but only `enabled` is used by the actual implementation

### TTS pipeline (agent output -> audio)

```
agent message_update ──▶ RegexTTSPolisher ──▶ TTSService ──▶ Python worker (Piper/Kokoro) ──▶ audio
```

- During `message_update` events, new text is extracted from the assistant message and buffered until a sentence boundary (`[.!?\n]$`)
- `flushTtsBuffer()` sends buffered text through `RegexTTSPolisher.polish()` then `TTSService.speak()`
- **`RegexTTSPolisher`** (`tts-polisher.ts`): purely deterministic regex, NO LLM calls, NO network. Strips thinking tags, code fences, markdown formatting, list/diff markers, humanizes identifiers (snake_case, kebab-case, camelCase, file paths), normalizes punctuation. Skips text under 30 chars.
- **`TTSService`** (`tts.ts`): spawns a persistent Python subprocess (`whisper_bot/tts_worker.py`) that preloads the TTS model once. Communicates via JSONL on stdin/stdout: `{"type":"speak","text":"...","speed":1.0}` → `{"type":"done"}`. Falls back to macOS `say` if Python unavailable.
- Supported TTS backends (Python): Piper (`tts/piper.py`) and Kokoro (`tts/kokoro.py`, `tts/streaming_kokoro.py`)
- Speech chains via `ttsSpeakChain` promise to avoid overlap while keeping sequential order

### Slash commands (defined in `slash-commands.ts`)

| Command | Handler location |
|---------|-----------------|
| `/stt` | `interactive-mode.ts` — `on`, `off`, `auto`, `parser`, `vad` subcommands |
| `/tts` | `interactive-mode.ts` — `on`, `off`, `polish` subcommands |
| `/speed` | `interactive-mode.ts` — set playback speed 0.5-3.0 step 0.25 |

### Keybindings (defined in `keybindings.ts`)

| Action | Default key | Description |
|--------|------------|-------------|
| `app.recording.toggle` | `ctrl+space` | Start/stop STT dictation |
| `app.tts.toggle` | (none) | Toggle TTS on/off |

### Services index (`packages/coding-agent/src/services/index.ts`)

Exports: `OllamaParser`, `STTService`, `TTSService`, `RegexTTSPolisher`, `TypeWhisperAPI`, and their types.

### Settings (defined in `settings-manager.ts`)

STT: `sttEnabled`, `sttAutoSubmit`, `sttParserEnabled`
TTS: `ttsEnabled`, `ttsSpeed`, `ttsPolisherEnabled`

### Whisper Bot (`packages/coding-agent/whisper_bot/`)

Python side of the TTS pipeline. Notable files:
- `tts/piper.py` — Piper TTS with speed control and multi-voice
- `tts/kokoro.py`, `tts/streaming_kokoro.py` — Kokoro TTS
- `tts/sentence_splitter.py` — smart sentence segmentation
- `tts_worker.py` — persistent worker process (JSONL stdin/stdout protocol)
- `parser/ollama.py` — Python Ollama parser (separate from TS version, calls Ollama HTTP API for qwen2.5-coder:3b). Used by the Python whisper_bot, NOT the TUI mode.

## Conversational Style

- Keep answers short and concise
- No emojis in commits, issues, PR comments, or code
- No fluff or cheerful filler text (e.g., "Thanks @user" not "Thanks so much @user!")
- Technical prose only, be direct
- When the user asks a question, answer it first before making edits or running implementation commands.
- When responding to user feedback or an analysis, explicitly say whether you agree or disagree before saying what you changed.

## Code Quality

- Read files in full before wide-ranging changes, before editing files you have not fully inspected, and when asked to investigate or audit. Do not rely on search snippets for broad changes.
- No `any` unless absolutely necessary.
- Inline single-line helpers that have only one call site.
- Check node_modules for external API types; don't guess.
- **No inline imports** (`await import()`, `import("pkg").Type`, dynamic type imports). Top-level imports only.
- Never remove or downgrade code to fix type errors from outdated deps; upgrade the dep instead.
- Use only erasable TypeScript syntax (Node strip-only mode) in code checked by the root config (`packages/*/src`, `packages/*/test`, `packages/coding-agent/examples`): no parameter properties, `enum`, `namespace`/`module`, `import =`, `export =`, or other constructs needing JS emit. Use explicit fields with constructor assignments.
- Always ask before removing functionality or code that appears intentional.
- Do not preserve backward compatibility unless the user asks for it.
- Never hardcode key checks (e.g. `matchesKey(keyData, "ctrl+x")`). Add defaults to `DEFAULT_EDITOR_KEYBINDINGS` or `DEFAULT_APP_KEYBINDINGS` so they stay configurable.
- Never modify `packages/ai/src/models.generated.ts` directly; update `packages/ai/scripts/generate-models.ts` instead, then regenerate. Including the resulting `models.generated.ts` diff is always OK, even if regeneration includes unrelated upstream model metadata changes.

## Commands

- After code changes (not docs): `npm run check` (full output, no tail). Fix all errors, warnings, and infos before committing. Does not run tests.
- Never run `npm run build` or `npm test` unless requested by the user.
- Never run the full vitest suite directly: it includes e2e tests that activate when endpoint/auth env vars are present. For all non-e2e tests, run `./test.sh` from the repo root. Otherwise run specific tests from the package root: `node ../../node_modules/vitest/dist/cli.js --run test/specific.test.ts`.
- If you create or modify a test file, run it and iterate on test or implementation until it passes.
- For `packages/coding-agent/test/suite/`, use `test/suite/harness.ts` + the faux provider. No real provider APIs, keys, or paid tokens.
- Put issue-specific regressions under `packages/coding-agent/test/suite/regressions/` named `<issue-number>-<short-slug>.test.ts`.
- For ad-hoc scripts, `write` them to a temp file (e.g. `/tmp`), run, edit if needed, remove when done. Don't embed multi-line scripts in `bash` commands.
- Never commit unless the user asks.

## Dependency and Install Security

- Treat npm dep and lockfile changes as reviewed code. Direct external deps stay pinned to exact versions.
- Hydrate/update locally with `npm install --ignore-scripts`; clean/CI-style with `npm ci --ignore-scripts`. Don't run lifecycle scripts unless the user asks.
- If dep metadata changes, refresh `package-lock.json` with `npm install --package-lock-only --ignore-scripts`.
- If `packages/coding-agent/npm-shrinkwrap.json` needs regen, run `node scripts/generate-coding-agent-shrinkwrap.mjs` (verify with `--check` or `npm run check`). New deps with lifecycle scripts require review and an explicit allowlist entry in that script; never add one silently.
- Pre-commit blocks lockfile commits unless `PI_ALLOW_LOCKFILE_CHANGE=1`. Don't bypass unless the user wants the lockfile change committed.

## Git

Multiple pi sessions may be running in this cwd at the same time, each modifying different files. Git operations that touch unstaged, staged, or untracked files outside your own changes will stomp on other sessions' work. Follow these rules:

Committing:

- Only commit files YOU changed in THIS session.
- Stage explicit paths (`git add <path1> <path2>`); never `git add -A` / `git add .`.
- Before committing, run `git status` and verify you are only staging your files.
- `packages/ai/src/models.generated.ts` may always be included alongside your files.
- Message format: `{feat,fix,docs}[(ai,tui,agent,coding-agent)]: <commit message> (optionally multiple lines)`. Message is informative and concise.

Never run (destroys other agents' work or bypasses checks):

- `git reset --hard`, `git checkout .`, `git clean -fd`, `git stash`, `git add -A`, `git add .`, `git commit --no-verify`.

If rebase conflicts occur:

- Resolve conflicts only in files you modified.
- If a conflict is in a file you did not modify, abort and ask the user.
- Never force push.

## Issues and PRs

See `CONTRIBUTING.md` for the contributor gate (auto-close workflows, `lgtm`/`lgtmi`, quality bar).

When reviewing PRs:

- Do not run `gh pr checkout`, `git switch`, or otherwise move the worktree to the PR branch unless the user explicitly asks.
- Use `gh pr view`, `gh pr diff`, `gh api`, and local `git show`/`git diff` against fetched refs to inspect PR metadata, commits, and patches without changing branches.
- If you need PR file contents, fetch/read them into temporary files or use `git show <ref>:<path>` without switching branches.

When creating issues:

- Add `pkg:*` labels for affected packages (`pkg:agent`, `pkg:ai`, `pkg:coding-agent`, `pkg:tui`); use all that apply.

When posting issue/PR comments:

- Write the comment to a temp file and post with `gh issue/pr comment --body-file` (never multi-line markdown via `--body`).
- Keep comments concise, technical, in the user's tone.
- End every AI-posted comment with the AI-generated disclaimer line specified by the originating prompt (e.g. `This comment is AI-generated by `/wr``).

When closing issues via commit:

- Include `fixes #<number>` or `closes #<number>` in the message so merging auto-closes the issue. For multiple issues, repeat the keyword per issue (`closes #1, closes #2`); a shared keyword (`closes #1, #2`) only closes the first.

## Testing pi Interactive Mode with tmux

Run the TUI in a controlled terminal (from the repo root):

```bash
tmux new-session -d -s pi-test -x 80 -y 24
tmux send-keys -t pi-test "./pi-test.sh" Enter
sleep 3 && tmux capture-pane -t pi-test -p     # capture after startup
tmux send-keys -t pi-test "your prompt here" Enter
tmux send-keys -t pi-test Escape               # special keys (also C-o for ctrl+o, etc.)
tmux kill-session -t pi-test
```

## Changelog

Location: `packages/*/CHANGELOG.md` (one per package).

Sections under `## [Unreleased]`: `### Breaking Changes` (API changes requiring migration), `### Added`, `### Changed`, `### Fixed`, `### Removed`.

Rules:

- All new entries go under `## [Unreleased]`. Read the full section first and append to existing subsections; never duplicate them.
- Released version sections (e.g. `## [0.12.2]`) are immutable; never modify them.

Attribution:

- Internal (from issues): `Fixed foo bar ([#123](https://github.com/earendil-works/pi-mono/issues/123))`
- External contributions: `Added feature X ([#456](https://github.com/earendil-works/pi-mono/pull/456) by [@username](https://github.com/username))`

## Releasing

**Lockstep versioning**: all packages share one version; every release updates all together. `patch` = fixes + additions, `minor` = breaking changes. No major releases.

1. **Update CHANGELOGs**: ask the user whether they ran the `/cl` prompt on the latest commit on `main`. If not, they must run `/cl` first to audit and update each package's `[Unreleased]` section before releasing.

2. **Local smoke test**: build an unpublished release and smoke test from outside the repo (so it can't resolve workspace files):
   ```bash
   npm run release:local -- --out /tmp/pi-local-release --force
   cd /tmp

   # Node package install smoke tests
   /tmp/pi-local-release/node/pi --help
   /tmp/pi-local-release/node/pi --version
   /tmp/pi-local-release/node/pi --list-models
   /tmp/pi-local-release/node/pi -p "Say exactly: ok"
   /tmp/pi-local-release/node/pi

   # Bun binary smoke tests
   /tmp/pi-local-release/bun/pi --help
   /tmp/pi-local-release/bun/pi --version
   /tmp/pi-local-release/bun/pi --list-models
   /tmp/pi-local-release/bun/pi -p "Say exactly: ok"
   /tmp/pi-local-release/bun/pi
   ```
   Verify both Node and Bun startup, model/account listing, interactive startup, and at least one real prompt with the intended default provider. The bare commands `/tmp/pi-local-release/node/pi` and `/tmp/pi-local-release/bun/pi` start interactive mode; run each in tmux, submit a prompt, and wait for the model reply before considering the interactive smoke test passed. Failures are release blockers unless the user explicitly accepts the risk.

3. **Run the release script**:
   ```bash
   PI_ALLOW_LOCKFILE_CHANGE=1 npm_config_min_release_age=0 npm run release:patch    # fixes + additions
   PI_ALLOW_LOCKFILE_CHANGE=1 npm_config_min_release_age=0 npm run release:minor    # breaking changes
   ```
   Use `npm_config_min_release_age=0` only for the release command. The repo's normal npm age gate can otherwise block the release lockfile refresh when the current workspace package version was published recently. Review any lockfile or shrinkwrap diffs the release creates before push.

   The release script bumps all package versions, updates changelogs, regenerates release artifacts, runs `npm run check`, commits `Release vX.Y.Z`, tags `vX.Y.Z`, adds fresh `## [Unreleased]` changelog sections, commits `Add [Unreleased] section for next cycle`, then pushes `main` and the tag. Do not rerun the release script after a tag was pushed.

4. **CI publishes npm packages**: pushing the `vX.Y.Z` tag triggers `.github/workflows/build-binaries.yml`. The `publish-npm` job uses npm trusted publishing through GitHub Actions OIDC with environment `npm-publish`; no local `npm publish`, `npm whoami`, OTP, or WebAuthn flow is required.

5. **If CI publish fails**: inspect the failed `publish-npm` job. The publish helper is idempotent and skips package versions already present on npm, so rerun the tag workflow after fixing CI or transient npm issues. Do not rerun `npm run release:patch` or `npm run release:minor` for the same version.

## User Override

If the user's instructions conflict with any rule in this document, ask for explicit confirmation before overriding. Only then execute their instructions.

## Agent skills

### Issue tracker

Issues live as GitHub Issues on `dantevannini30-cell/Pi-Speech`. Uses the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default five-role vocabulary (needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix). See `docs/agents/triage-labels.md`.

### Domain docs

Multi-context layout. Root `CONTEXT.md` provides a high-level overview and domain glossary; `CONTEXT-MAP.md` points to per-package `CONTEXT.md` files in `packages/coding-agent/`, `packages/ai/`, `packages/agent/`, and `packages/tui/`. See `docs/agents/domain.md`.
