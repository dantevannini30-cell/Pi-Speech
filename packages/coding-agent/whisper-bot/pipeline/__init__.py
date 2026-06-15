"""Pipeline orchestrator — wires STT → parser → agent → TTS + display together.

Uses an ``asyncio.Queue`` chain between stages with a single unified event bus
for display and side-effects.
"""

from __future__ import annotations

import asyncio
import collections.abc
import sys
from enum import Enum, auto
from typing import Any

from whisper_bot.agent import AgentProvider
from whisper_bot.agent.pi import PiAgentProvider
from whisper_bot.agent.pi_rpc import PiRpcAgentProvider
from whisper_bot.bootstrap import check_dependencies, print_status
from whisper_bot.config import load_config
from whisper_bot.debug import log as _debug
from whisper_bot.debug import log_exc as _debug_exc
from whisper_bot.display import Display
from whisper_bot.events import EventBus
from whisper_bot.events.types import (
    AgentDone,
    AgentToken,
    InputCommand,
    ParsingDone,
    ParsingStarted,
    PipelineError,
    STTDone,
    STTStarted,
    STTTranscribing,
    TTSDone,
    TTSSpeaking,
    TTSSentenceStart,
    TTSSentenceDone,
    ToolExecutionEnd,
    ToolExecutionStart,
    UserTextInput,
)
from whisper_bot.input import InputHandler
from whisper_bot.parser import ParserProvider
from whisper_bot.parser.ollama import OllamaParser
from whisper_bot.stt import STTProvider
from whisper_bot.stt.typewhisper import TypeWhisperProvider
from whisper_bot.tts import TTSProvider
from whisper_bot.tts.kokoro import KokoroTTS
from whisper_bot.tts.piper import PiperTTS, StreamingPiperTTS
from whisper_bot.tts.sentence_splitter import SentenceSplitter
from whisper_bot.tts.streaming_kokoro import StreamingKokoroTTS
from whisper_bot.audio import AudioFeedback


class PipelineState(Enum):
    """States for the pipeline state machine."""
    IDLE = auto()
    LISTENING = auto()
    TRANSCRIBING = auto()
    PARSING = auto()
    AGENT_STREAMING = auto()
    SPEAKING = auto()



class Pipeline:
    """Orchestrates the STT → parser → agent → TTS + display pipeline."""

    def __init__(
        self,
        config: dict[str, Any],
        event_bus: EventBus,
        stt: STTProvider,
        parser_provider: ParserProvider,
        agent: AgentProvider,
        tts: TTSProvider,
        display: Display,
        shutdown_event: asyncio.Event,
        audio_feedback: AudioFeedback | None = None,
        streaming_tts: StreamingKokoroTTS | StreamingPiperTTS | None = None,
        idle_event: asyncio.Event | None = None,
    ) -> None:
        self.config = config
        self.event_bus = event_bus
        self.stt = stt
        self.parser = parser_provider
        self.agent = agent
        self.tts = tts
        self.display = display
        self._shutdown = shutdown_event
        self.audio_feedback = audio_feedback
        self.streaming_tts = streaming_tts
        self.idle_event = idle_event or asyncio.Event()
        self.idle_event.set()  # start idle

        self.state = PipelineState.IDLE
        self._tasks: list[asyncio.Task[Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Start the pipeline and process events until quit."""
        _debug("[DEBUG pipeline] Pipeline.run() started")
        # Initial prompt is shown by InputHandler after it drains stdin

        async for event in self.event_bus.subscribe():
            if self._shutdown.is_set():
                _debug("[DEBUG pipeline] Shutdown event set — calling agent.shutdown() and breaking")
                await self.agent.shutdown()
                break

            event_type = type(event).__name__
            _debug(f"[DEBUG pipeline] Event received: {event_type}  (state={self.state.name})")

            if isinstance(event, InputCommand):
                _debug(f"[DEBUG pipeline] Handling InputCommand: {event.action}")
                await self._handle_command(event)
            elif isinstance(event, STTStarted):
                self.state = PipelineState.LISTENING
                if self.audio_feedback:
                    asyncio.create_task(self.audio_feedback.play("start"))
                _debug("[DEBUG pipeline] State -> LISTENING")
            elif isinstance(event, STTTranscribing):
                self.state = PipelineState.TRANSCRIBING
                if self.audio_feedback:
                    asyncio.create_task(self.audio_feedback.play("stop"))
                _debug("[DEBUG pipeline] State -> TRANSCRIBING")
            elif isinstance(event, STTDone):
                self.state = PipelineState.PARSING
                _debug(f"[DEBUG pipeline] STTDone received, text={event.text!r}, spawning _run_parse task")
                asyncio.create_task(self._run_parse(event.text))
            elif isinstance(event, UserTextInput):
                self.state = PipelineState.AGENT_STREAMING
                _debug(f"[DEBUG pipeline] UserTextInput received, text={event.text!r}, spawning _run_agent directly")
                asyncio.create_task(self._run_agent(event.text))
            elif isinstance(event, ParsingStarted):
                pass  # status already shown by _stop_listening
            elif isinstance(event, ParsingDone):
                self.state = PipelineState.AGENT_STREAMING
                _debug(f"[DEBUG pipeline] ParsingDone: original={event.original!r}  cleaned={event.cleaned!r}")
                self.display.replace_line(f"  You: {event.cleaned}")
                _debug(f"[DEBUG pipeline] Spawning _run_agent task with cleaned text ({len(event.cleaned)} chars)")
                asyncio.create_task(self._run_agent(event.cleaned))
            elif isinstance(event, AgentToken):
                self.display.stream_token(event.text)
            elif isinstance(event, AgentDone):
                self.state = PipelineState.SPEAKING
                _debug(f"[DEBUG pipeline] AgentDone: full_text ({len(event.full_text)} chars)")
                _debug(f"[DEBUG pipeline] AgentDone first 200 chars: {event.full_text[:200]!r}")
                self.display.newline()
                if self.audio_feedback:
                    asyncio.create_task(self.audio_feedback.play("chime"))
                if not self.streaming_tts:  # streaming already handled in _run_agent
                    _debug("[DEBUG pipeline] Spawning _run_tts task")
                    asyncio.create_task(self._run_tts(event.full_text))
            elif isinstance(event, TTSSpeaking):
                _debug("[DEBUG pipeline] TTS speaking...")
                self.display.show_status("  [speaking...]")
            elif isinstance(event, TTSDone):
                self.state = PipelineState.IDLE
                _debug("[DEBUG pipeline] TTS done, returning to IDLE")
                self.display.show_status("  [done]")
                self.idle_event.set()
                # Prompt shown by InputHandler after draining stdin
            elif isinstance(event, PipelineError):
                _debug(f"[DEBUG pipeline] PipelineError: stage={event.stage} message={event.message!r} fatal={event.fatal}")
                if event.fatal and self.audio_feedback:
                    asyncio.create_task(self.audio_feedback.play("error"))
                await self._handle_error(event)

    # ------------------------------------------------------------------
    # Agent lifecycle helpers
    # ------------------------------------------------------------------

    async def restart_agent(self) -> None:
        """Shut down the current agent subprocess so it re-spawns on the
        next turn with any config changes (e.g. a new model)."""
        _debug("[DEBUG pipeline] restart_agent() called — shutting down agent")
        await self.agent.shutdown()
        _debug("[DEBUG pipeline] Agent shut down — will re-spawn on next process() call")

    async def stop_speaking(self) -> None:
        """Interrupt TTS playback immediately."""
        _debug("[DEBUG pipeline] stop_speaking() called — cancelling TTS worker")
        if self.streaming_tts is not None:
            await self.streaming_tts.shutdown()
        _debug("[DEBUG pipeline] TTS worker cancelled")

    # ------------------------------------------------------------------
    # Command handling
    # ------------------------------------------------------------------

    async def _handle_command(self, cmd: InputCommand) -> None:
        if cmd.action == "listen":
            await self._start_listening()
        elif cmd.action == "stop":
            await self._stop_listening()
        elif cmd.action == "quit":
            _debug("[DEBUG pipeline] Quit command received — shutting down agent and setting shutdown event")
            await self.agent.shutdown()
            self._shutdown.set()
        elif cmd.action == "help":
            self._show_help()

    async def _start_listening(self) -> None:
        _debug(f"[DEBUG pipeline] _start_listening called, current state={self.state.name}")
        if self.state != PipelineState.IDLE:
            self.display.show_status("  [already recording — type 'stop' to end]")
            _debug(f"[DEBUG pipeline] Refusing to start: state={self.state.name} != IDLE")
            return

        self.display.show_status("  [recording started — type 'stop' to end]")
        try:
            await self.stt.start()
            _debug("[DEBUG pipeline] STT.start() completed successfully")
        except PipelineError as e:
            _debug(f"[DEBUG pipeline] STT.start() failed: {e}")
            self.state = PipelineState.IDLE
            self.display.show_status("  [STT failed — TypeWhisper not running?]")
            self.idle_event.set()

    async def _stop_listening(self) -> None:
        _debug(f"[DEBUG pipeline] _stop_listening called, current state={self.state.name}")
        if self.state != PipelineState.LISTENING:
            self.display.show_status("  [not currently recording — type 'listen' to start]")
            _debug(f"[DEBUG pipeline] Refusing to stop: state={self.state.name} != LISTENING")
            return

        self.display.show_status("  [transcribing...]")
        try:
            await self.stt.stop()
            _debug("[DEBUG pipeline] STT.stop() completed successfully")
        except PipelineError as e:
            _debug(f"[DEBUG pipeline] STT.stop() failed: {e}")
            self.state = PipelineState.IDLE
            self.display.show_status("  [transcription failed]")
            self.idle_event.set()

        # STT.stop() already emitted STTDone; it flows through the event loop.

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    async def _run_parse(self, raw_text: str) -> None:
        _debug(f"[DEBUG pipeline] _run_parse starting with raw_text ({len(raw_text)} chars): {raw_text!r}")

        # If parser is disabled, pass raw STT straight to the agent
        if self.config.get("parser", {}).get("skip", False):
            _debug("[DEBUG pipeline] Parser skip=true — passing raw STT directly to agent")
            # replace_line is handled by the event loop when ParsingDone is received below
            self.state = PipelineState.AGENT_STREAMING
            await self.event_bus.emit(ParsingDone(original=raw_text, cleaned=raw_text))
            return

        try:
            cleaned = await self.parser.parse(raw_text)
            _debug(f"[DEBUG pipeline] _run_parse completed, cleaned ({len(cleaned)} chars): {cleaned!r}")
        except PipelineError as e:
            # Parser errors are recoverable — pass raw text to agent
            _debug(f"[DEBUG pipeline] Parser error (non-fatal): {e.message}")
            self.display.show_status(f"  [parser: {e.message}]")
            self.display.replace_line(f"  You: {raw_text}")
            self.state = PipelineState.AGENT_STREAMING
            _debug(f"[DEBUG pipeline] Falling back to raw text for agent (since parser failed)")
            asyncio.create_task(self._run_agent(raw_text))
            return
        except Exception as e:
            # Catch anything unexpected so it doesn't get swallowed by the task
            _debug(f"[DEBUG pipeline] UNEXPECTED exception in _run_parse: {type(e).__name__}: {e}")
            self.display.show_status(f"  [parser: unexpected error: {e}]")
            self.display.replace_line(f"  You: {raw_text}")
            self.state = PipelineState.AGENT_STREAMING
            asyncio.create_task(self._run_agent(raw_text))
            return

        # ParsingDone event already emitted by the provider; state transitions
        # via the event loop.

    @staticmethod
    def _tokenize_for_display(text: str) -> list[str]:
        """Split *text* into word-level chunks for smooth TUI streaming.

        The agent may yield large text blocks (entire paragraphs).  Splitting
        them into word-level pieces before emitting ``AgentToken`` events lets
        the TUI update incrementally, creating a visible typing effect.
        """
        if not text:
            return []
        words = text.split(" ")
        if len(words) <= 1:
            return [text]
        chunks: list[str] = []
        for i, word in enumerate(words):
            if i < len(words) - 1:
                chunks.append(word + " ")
            else:
                chunks.append(word)
        return chunks

    async def _run_agent(self, text: str) -> None:
        _debug(f"[DEBUG pipeline] _run_agent starting with text ({len(text)} chars): {text[:200]!r}")
        # Show label — no trailing newline, first token streams right after
        self.display.show_agent_label()
        token_count = 0

        # Streaming TTS: set up sentence splitter for incremental playback
        splitter = SentenceSplitter() if self.streaming_tts else None

        # Collect full response text to pass with AgentDone
        full_text_parts: list[str] = []

        try:
            async for token in self.agent.process(text):
                token_count += 1
                full_text_parts.append(token)
                if token_count <= 3:
                    _debug(f"[DEBUG pipeline] Agent token #{token_count}: {token!r}")

                # Split large blocks into word-level chunks for smooth TUI
                display_chunks = self._tokenize_for_display(token)
                for chunk in display_chunks:
                    await self.event_bus.emit(AgentToken(text=chunk))

                # Streaming TTS: feed the *full* token (not chunks) to the
                # sentence splitter so sentence boundaries are preserved.
                if self.streaming_tts and splitter:
                    for sentence in splitter.feed(token):
                        await self.streaming_tts.speak_sentence(sentence)
        except PipelineError as e:
            _debug(f"[DEBUG pipeline] _run_agent caught PipelineError: fatal={e.fatal} message={e.message!r}")
            if e.fatal:
                self.display.show_status(f"  [agent error: {e.message}]")
                self.state = PipelineState.IDLE
                self.idle_event.set()
                _debug("[DEBUG pipeline] Agent error was fatal, returning to IDLE")
                return
        except Exception as e:
            # CRITICAL: catch anything so asyncio task doesn't swallow it silently
            _debug(f"[DEBUG pipeline] UNEXPECTED exception in _run_agent: {type(e).__name__}: {e}")
            _debug_exc()
            self.display.show_status(f"  [agent error: {e}]")
            self.state = PipelineState.IDLE
            self.idle_event.set()
            return

        # ---- Agent completed normally (possibly with 0 tokens) ----
        full_text = "".join(full_text_parts)
        _debug(f"[DEBUG pipeline] Agent streamed {token_count} tokens, emitting AgentDone")
        await self.event_bus.emit(AgentDone(full_text=full_text))

        # Agent done — emit speaking status and flush remaining buffer
        if self.streaming_tts and splitter:
            await self.event_bus.emit(TTSSpeaking())
            for sentence in splitter.flush():
                await self.streaming_tts.speak_sentence(sentence)
            # Wait for all queued sentences to finish playing
            await self.streaming_tts.wait()
            await self.event_bus.emit(TTSDone())

        _debug(f"[DEBUG pipeline] _run_agent completed, streamed {token_count} tokens")

    async def _run_tts(self, text: str) -> None:
        _debug(f"[DEBUG pipeline] _run_tts starting with text ({len(text)} chars)")
        _debug(f"[DEBUG pipeline] TTS enabled in config: {self.config.get('tts', {}).get('enabled', True)}")
        try:
            await self.tts.speak(text)
            _debug("[DEBUG pipeline] _run_tts completed successfully")
        except PipelineError as e:
            _debug(f"[DEBUG pipeline] _run_tts caught PipelineError: {e.message!r}")
            self.display.show_status(f"  [TTS warning: {e.message}]")
            self.state = PipelineState.IDLE
            self.idle_event.set()
            return
        except Exception as e:
            _debug(f"[DEBUG pipeline] UNEXPECTED exception in _run_tts: {type(e).__name__}: {e}")
            _debug_exc()
            self.state = PipelineState.IDLE
            self.idle_event.set()
            return

        # TTSDone emitted by provider; back to IDLE in event loop.

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------

    async def _handle_error(self, error: PipelineError) -> None:
        _debug(f"[DEBUG pipeline] _handle_error: stage={error.stage} fatal={error.fatal} message={error.message!r}")
        if error.fatal:
            self.display.show_status(f"  [error: {error.message}]")
            self.state = PipelineState.IDLE
            self.idle_event.set()
        else:
            self.display.show_status(f"  [warning: {error.message}]")

    # ------------------------------------------------------------------
    # Help
    # ------------------------------------------------------------------

    def _show_help(self) -> None:
        self.display.show_status("  Commands:")
        self.display.show_status("    listen  — start recording")
        self.display.show_status("    stop    — stop recording and transcribe")
        self.display.show_status("    quit    — exit")
        self.display.show_status("    help    — show this message")
        # Prompt shown by InputHandler after draining stdin


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

async def create_pipeline(
    shutdown_event: asyncio.Event,
    event_bus: EventBus | None = None,
    *,
    skip_parser: bool = False,
    idle_event: asyncio.Event | None = None,
    display: Display | None = None,
    progress_callback: collections.abc.Callable[[str, str, float], Awaitable[None]] | None = None,
) -> Pipeline:
    """Create a fully wired pipeline from config.

    Parameters
    ----------
    shutdown_event:
        Fired when the pipeline should shut down.
    event_bus:
        Shared event bus.  Created fresh if omitted.
    skip_parser:
        If True, pass raw STT output directly to the agent without LLM parsing.
    idle_event:
        Fired when the pipeline is idle and ready for new input.
    display:
        Custom ``Display`` instance for routing output (e.g. to TUI widgets).
        A default ``Display`` is created if omitted.
    progress_callback:
        Optional async callable ``(provider, status, pct)`` for UI progress
        during warmup.  *provider* is one of ``\"stt\"``, ``\"pi\"``, ``\"tts\"``.
    """
    _debug("[DEBUG pipeline] create_pipeline() called")
    if skip_parser:
        _debug("[DEBUG pipeline] skip_parser=True — raw STT will be passed directly to agent")
    config = load_config()
    bus = event_bus or EventBus()
    display = display or Display()

    # CLI flag overrides config file
    if skip_parser:
        config.setdefault("parser", {})["skip"] = True

    _debug(f"[DEBUG pipeline] Config: stt.provider=typewhisper parser.provider=ollama agent.provider=pi tts.provider={config.get('tts', {}).get('provider', 'unknown')}")
    _debug(f"[DEBUG pipeline] Ollama endpoint: {config.get('parser', {}).get('endpoint', 'default')} model: {config.get('parser', {}).get('model', 'default')}")
    _debug(f"[DEBUG pipeline] TTS enabled: {config.get('tts', {}).get('enabled', True)} voice: {config.get('tts', {}).get('voice', 'default')}")

    stt: STTProvider = TypeWhisperProvider(config, bus)
    parser_provider: ParserProvider = OllamaParser(config, bus)

    # Select agent provider based on mode (rpc = persistent subprocess, oneshot = spawn per turn)
    agent_mode = config.get("agent", {}).get("mode", "rpc")
    if agent_mode == "rpc":
        _debug("[DEBUG pipeline] Using PiRpcAgentProvider (RPC mode — persistent subprocess)")
        agent: AgentProvider = PiRpcAgentProvider(config, bus)
    else:
        _debug("[DEBUG pipeline] Using PiAgentProvider (one-shot mode — spawn per turn)")
        agent: AgentProvider = PiAgentProvider(config, bus)

    # Audio feedback (earcons)
    audio_feedback = AudioFeedback(config)

    # TTS provider — select by name, then streaming or blocking
    tts_provider = config.get("tts", {}).get("provider", "piper")
    tts_streaming = config.get("tts", {}).get("streaming", True)
    streaming_tts: StreamingKokoroTTS | StreamingPiperTTS | None = None

    if tts_provider == "piper":
        if tts_streaming:
            _debug("[DEBUG pipeline] Using StreamingPiperTTS (streaming mode)")
            streaming_tts = StreamingPiperTTS(config, bus)
            tts = streaming_tts
        else:
            _debug("[DEBUG pipeline] Using PiperTTS (blocking mode)")
            tts = PiperTTS(config, bus)
    else:
        # Default to Kokoro for backward compat
        if tts_streaming:
            _debug("[DEBUG pipeline] Using StreamingKokoroTTS (streaming mode)")
            streaming_tts = StreamingKokoroTTS(config, bus)
            tts = streaming_tts
        else:
            _debug("[DEBUG pipeline] Using KokoroTTS (blocking mode)")
            tts = KokoroTTS(config, bus)

    # ------------------------------------------------------------------
    # Warm up — eagerly start persistent subprocesses so they're ready
    # before the first turn.  Blocks startup until all providers signal
    # readiness so the user never experiences a cold start after seeing
    # the TUI.
    # ------------------------------------------------------------------
    _debug("[DEBUG pipeline] Warming up providers (blocking)...")

    # 1. STT — ensure TypeWhisper is running (auto-launch if needed)
    if hasattr(stt, '_ensure_running'):
        if progress_callback is not None:
            await progress_callback("stt", "launching...", 0.1)
        stt_ok = await stt._ensure_running()
        if progress_callback is not None:
            await progress_callback("stt", "ready" if stt_ok else "failed", 1.0)
        if not stt_ok:
            _debug("[DEBUG pipeline] TypeWhisper failed to start")
    else:
        if progress_callback is not None:
            await progress_callback("stt", "ready", 1.0)

    # 2. Pi agent (persistent RPC subprocess)
    if progress_callback is not None:
        await progress_callback("pi", "spawning agent...", 0.2)
    if hasattr(agent, "warmup"):
        try:
            await agent.warmup()
        except Exception:
            _debug("[DEBUG pipeline] Agent warmup failed (will lazy-spawn later)")
    if progress_callback is not None:
        await progress_callback("pi", "ready", 1.0)

    # 3. TTS (persistent Kokoro subprocess pool)
    if streaming_tts is not None:
        if progress_callback is not None:
            await progress_callback("tts", "loading model (30-60s)...", 0.1)
        try:
            await streaming_tts.warmup(progress_callback=progress_callback)
        except Exception:
            _debug("[DEBUG pipeline] TTS warmup failed (will lazy-spawn later)")

    _debug("[DEBUG pipeline] Warmup complete")

    return Pipeline(
        config, bus, stt, parser_provider, agent, tts, display, shutdown_event,
        audio_feedback=audio_feedback,
        streaming_tts=streaming_tts,
        idle_event=idle_event,
    )


async def run_pipeline(*, skip_parser: bool = False) -> None:
    """Run dependency checks and start the pipeline."""
    _debug("[DEBUG pipeline] run_pipeline() started")
    deps = check_dependencies()
    print_status(deps)

    for dep_name, dep_ok in deps.items():
        _debug(f"[DEBUG pipeline] Dependency check: {dep_name} = {dep_ok}")

    if not deps["pi"]:
        print("Fatal: Pi agent CLI is required.", file=sys.stderr)
        sys.exit(1)

    shutdown_event = asyncio.Event()
    event_bus = EventBus()

    _debug("[DEBUG pipeline] Creating pipeline and input handler...")
    idle_event = asyncio.Event()
    idle_event.set()
    pipeline = await create_pipeline(shutdown_event, event_bus, skip_parser=skip_parser, idle_event=idle_event)
    input_handler = InputHandler(event_bus, shutdown_event, pipeline_idle=idle_event)

    # Run input handler and pipeline concurrently; on quit, both
    # tasks see the shutdown event and exit.
    _debug("[DEBUG pipeline] Entering asyncio.TaskGroup...")
    async with asyncio.TaskGroup() as tg:
        tg.create_task(input_handler.run())
        tg.create_task(pipeline.run())
    _debug("[DEBUG pipeline] TaskGroup exited (both tasks completed)")