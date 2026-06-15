"""Textual TUI application for whisper-bot.

The pipeline runs as a background worker and routes all display output to TUI
widgets via a custom ``TuiDisplay``.  Spacebar toggles recording on/off
(handled via ``on_key``, not BINDINGS, so we can prevent the default space
action).  The ``Input`` bar accepts ``/model`` commands to switch the Pi
backend model at runtime (e.g. ``/model deepseek/deepseek-v4-flash``).
"""

from __future__ import annotations

import asyncio
import json
import os

from textual.app import App, ComposeResult
from textual.events import Key
from textual.screen import Screen
from textual.widgets import Header, Input, ProgressBar, Static
from textual.containers import Horizontal

from whisper_bot.events import EventBus
from whisper_bot.events.types import (
    InputCommand,
    STTDone,
    STTStarted,
    ToolExecutionEnd,
    ToolExecutionStart,
    UserTextInput,
)
from whisper_bot.pipeline import create_pipeline
from whisper_bot.tui.chat_pane import ChatPane
from whisper_bot.tui.status_bar import WhisperStatusBar
from whisper_bot.tui.tool_pane import ToolPane
from whisper_bot.tui.tui_display import TuiDisplay


PI_SETTINGS_PATH = os.path.expanduser("~/.pi/agent/settings.json")


def _read_pi_settings() -> dict:
    """Read Pi's settings.json, returning an empty dict if missing."""
    try:
        with open(PI_SETTINGS_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _write_pi_settings(settings: dict) -> None:
    """Write Pi's settings.json atomically."""
    os.makedirs(os.path.dirname(PI_SETTINGS_PATH), exist_ok=True)
    tmp = PI_SETTINGS_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(settings, f, indent=2)
        f.write("\n")
    os.replace(tmp, PI_SETTINGS_PATH)


class LoadingScreen(Screen):
    """Full-screen loading indicator shown while providers warm up.

    Shows per-provider progress (STT, Pi, TTS) with ProgressBar widgets.
    Removed once all providers signal readiness.
    """

    CSS = """
    LoadingScreen {
        align: center middle;
    }
    #loading-title {
        text-style: bold;
        width: 100%;
        text-align: center;
        margin-bottom: 2;
    }
    .provider-row {
        height: 3;
        width: 64;
        margin-bottom: 0;
    }
    .provider-label {
        width: 6;
        text-style: bold;
        padding-top: 0;
    }
    ProgressBar {
        width: 36;
    }
    .status-text {
        width: 22;
        padding-top: 0;
    }
    """

    def compose(self) -> ComposeResult:
        yield Static("🔊 Initializing whisper-bot...", id="loading-title")
        with Horizontal(classes="provider-row"):
            yield Static("STT", classes="provider-label")
            yield ProgressBar(total=100, id="stt-bar")
            yield Static("waiting...", id="stt-status", classes="status-text")
        with Horizontal(classes="provider-row"):
            yield Static("Pi", classes="provider-label")
            yield ProgressBar(total=100, id="pi-bar")
            yield Static("waiting...", id="pi-status", classes="status-text")
        with Horizontal(classes="provider-row"):
            yield Static("TTS", classes="provider-label")
            yield ProgressBar(total=100, id="tts-bar")
            yield Static("waiting...", id="tts-status", classes="status-text")

    def update_provider(self, provider: str, status: str, pct: float) -> None:
        """Update progress for one provider row.

        Args:
            provider:  ``\"stt\"``, ``\"pi\"``, or ``\"tts\"``
            status:    Short human-readable status text (e.g. ``\"ready\"``)
            pct:       Float 0.0–1.0 for the progress bar
        """
        try:
            bar = self.query_one(f"#{provider}-bar", ProgressBar)
            label = self.query_one(f"#{provider}-status", Static)
            bar.progress = int(pct * 100)
            label.update(status)
        except Exception:
            pass  # widgets may not be mounted yet


class WhisperBotApp(App[None]):
    """Textual TUI for the whisper-bot pipeline."""

    CSS = """
    Screen {
        layout: horizontal;
    }

    ChatPane {
        width: 1fr;
        height: 100%;
    }

    ToolPane {
        width: 40;
        height: 100%;
    }
    ToolPane.hidden {
        display: none;
    }

    /* Command input bar — sits above the status bar */
    #cmd-input {
        dock: bottom;
        height: 1;
        margin-bottom: 1;
        background: $surface;
        color: $text;
    }

    WhisperStatusBar {
        dock: bottom;
        height: 1;
        background: $panel;
        color: $text-muted;
    }
    """

    BINDINGS = [
        ("ctrl+t", "toggle_tts"),
        ("ctrl+e", "toggle_tools"),
        ("ctrl+c", "quit"),
        ("ctrl+l", "clear_session"),
        ("escape", "focus_input"),
    ]

    def __init__(self, skip_parser: bool = False) -> None:
        super().__init__()
        self._skip_parser = skip_parser
        self._pipeline = None
        self._event_bus: EventBus | None = None
        self._shutdown_event: asyncio.Event | None = None
        self._recording = False
        self._loading_screen: LoadingScreen | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield ChatPane()
        yield ToolPane(classes="hidden")
        yield Input(id="cmd-input", placeholder="type a message or /model <name> ...")
        yield WhisperStatusBar()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_focus_input(self) -> None:
        """Focus the command input bar (Esc)."""
        self.query_one("#cmd-input", Input).focus()

    async def on_mount(self) -> None:
        """Push loading screen and start pipeline warmup."""
        self._loading_screen = LoadingScreen()
        await self.push_screen(self._loading_screen)
        self.run_worker(self._run_pipeline(), exclusive=True)

    # ------------------------------------------------------------------
    # Command input handler — /model commands + free-text to agent
    # ------------------------------------------------------------------

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle the command input bar submission."""
        text = event.value.strip()
        cmd_input = event.input
        cmd_input.clear()

        if not text:
            return

        if text.startswith("/"):
            await self._handle_command(text)
        else:
            await self._send_to_pipeline(text)

    async def _handle_command(self, text: str) -> None:
        """Parse and execute slash commands."""
        parts = text.split(maxsplit=1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if command == "/model":
            await self._cmd_model(args)
        elif command == "/help":
            self._cmd_help()
        else:
            self.query_one(ChatPane).add_status(
                f"Unknown command: {command}. Try /model or /help"
            )

    async def _cmd_model(self, args: str) -> None:
        """Handle ``/model`` and ``/model <name>``."""
        chat = self.query_one(ChatPane)

        if args:
            # Set model: update pi's settings.json and restart the agent
            new_model = args.strip()

            # Read current pi settings
            settings = _read_pi_settings()
            old_model = settings.get("defaultModel", "?")
            old_provider = settings.get("defaultProvider", "?")

            # Infer provider from model name (openrouter models have a "/")
            if "/" in new_model and not new_model.startswith("openrouter/"):
                new_provider = "openrouter"
                full_model = new_model
            else:
                new_provider = old_provider  # keep existing provider
                full_model = new_model

            settings["defaultProvider"] = new_provider
            settings["defaultModel"] = full_model
            _write_pi_settings(settings)

            chat.add_status(
                f"Model: {old_provider}/{old_model} → {new_provider}/{full_model}"
            )

            # Restart the persistent Pi subprocess so it picks up the change
            if self._pipeline is not None:
                await self._pipeline.restart_agent()

            chat.add_status("Agent restarted — next turn uses the new model")
        else:
            # Show current model
            settings = _read_pi_settings()
            prov = settings.get("defaultProvider", "?")
            model = settings.get("defaultModel", "?")
            chat.add_status(f"Current model: {prov}/{model}")

    def _cmd_help(self) -> None:
        """Show available commands."""
        chat = self.query_one(ChatPane)
        chat.add_status("Commands:")
        chat.add_status("  /model            — show current model")
        chat.add_status("  /model <name>     — switch model (restarts agent)")
        chat.add_status("  /help             — this message")
        chat.add_status("  Esc              — focus input bar")
        chat.add_status("  Ctrl+L           — clear chat")
        chat.add_status("  Ctrl+T           — toggle TTS")
        chat.add_status("  Ctrl+E           — toggle tool pane")

    async def _send_to_pipeline(self, text: str) -> None:
        """Send typed text straight to the agent, bypassing STT + parser."""
        if self._event_bus is None:
            return
        chat = self.query_one(ChatPane)
        chat.add_user_message(text)
        await self._event_bus.emit(UserTextInput(text=text))

    # ------------------------------------------------------------------
    # Key handlers — spacebar toggles recording on/off
    # ------------------------------------------------------------------

    async def on_key(self, event: Key) -> None:
        """Handle key events — space toggles recording, or interrupts TTS."""
        if event.key != "space":
            return
        event.stop()

        if self._event_bus is None:
            return

        # If TTS is speaking — stop it and start listening
        if self._pipeline is not None and self._pipeline.state.name == "SPEAKING":
            self._recording = True
            self.query_one(WhisperStatusBar).recording = True
            await self._stop_tts_and_listen()
            return

        if self._recording:
            # Stop recording
            self._recording = False
            self.query_one(WhisperStatusBar).recording = False
            await self._event_bus.emit(InputCommand(action="stop"))
        else:
            # Start recording
            self._recording = True
            self.query_one(WhisperStatusBar).recording = True
            await self._event_bus.emit(InputCommand(action="listen"))

    async def _stop_tts_and_listen(self) -> None:
        """Interrupt TTS and immediately start a new recording."""
        if self._pipeline is not None:
            await self._pipeline.stop_speaking()
        # Wait until pipeline reaches IDLE (TTSDone processed)
        for _ in range(100):
            if self._pipeline is None or self._pipeline.state.name == "IDLE":
                break
            await asyncio.sleep(0.05)
        if self._event_bus is not None:
            await self._event_bus.emit(InputCommand(action="listen"))

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def action_toggle_tts(self) -> None:
        """Toggle TTS on/off."""
        status_bar = self.query_one(WhisperStatusBar)
        status_bar.tts_enabled = not status_bar.tts_enabled
        if self._pipeline is not None:
            self._pipeline.config.setdefault("tts", {})[
                "enabled"
            ] = status_bar.tts_enabled

    def action_toggle_tools(self) -> None:
        """Show or hide the tool execution pane."""
        self.query_one(ToolPane).toggle_visible()

    def action_clear_session(self) -> None:
        """Clear the chat pane."""
        self.query_one(ChatPane).clear()

    async def action_quit(self) -> None:
        """Shut down the pipeline and exit the app."""
        if self._shutdown_event is not None:
            self._shutdown_event.set()
        # Give the pipeline a moment to shut down its subprocess
        await asyncio.sleep(0.5)
        self.exit()

    # ------------------------------------------------------------------
    # Pipeline worker
    # ------------------------------------------------------------------

    async def _run_pipeline(self) -> None:
        """Run the pipeline in a background worker.

        This replaces the stdin-based ``InputHandler + Pipeline`` pattern
        from the CLI entry point.  Spacebar events are routed via
        ``_on_key_down`` / ``_on_key_up`` instead.
        """
        from whisper_bot.bootstrap import check_dependencies, print_status

        deps = check_dependencies()
        print_status(deps)

        if not deps.get("pi"):
            self.notify("Pi agent CLI is required", severity="error")
            return

        # Capture widget references for use from sync subscriber and display
        chat_pane = self.query_one(ChatPane)
        status_bar = self.query_one(WhisperStatusBar)
        tool_pane = self.query_one(ToolPane)
        tui_display = TuiDisplay(chat_pane, status_bar)

        shutdown_event = asyncio.Event()
        self._shutdown_event = shutdown_event
        event_bus = EventBus()
        self._event_bus = event_bus

        # Always idle in TUI mode — no stdin InputHandler runs
        idle_event = asyncio.Event()
        idle_event.set()

        # Sync handler for bus events — fires synchronously inside emit().
        # Since all tasks run in Textual's event loop, widget methods can be
        # called directly (no thread-safety wrapper needed).
        def on_bus_event(event):
            if isinstance(event, STTStarted):
                self._recording = True
                status_bar.recording = True
            elif isinstance(event, STTDone):
                self._recording = False
                status_bar.recording = False
            elif isinstance(event, ToolExecutionStart):
                tool_pane.add_tool_call(
                    event.tool_name,
                    event.args or "",
                )
            elif isinstance(event, ToolExecutionEnd):
                tool_pane.finish_tool_call(
                    event.tool_call_id,
                    event.result or "",
                )

        event_bus.subscribe_sync(on_bus_event)

        # Progress callback for the loading screen
        loading = self._loading_screen

        async def on_progress(provider: str, status: str, pct: float) -> None:
            loading.update_provider(provider, status, pct)

        pipeline = await create_pipeline(
            shutdown_event,
            event_bus,
            skip_parser=self._skip_parser,
            display=tui_display,
            idle_event=idle_event,
            progress_callback=on_progress,
        )
        self._pipeline = pipeline

        # Mark STT and Pi as ready (their warmup is quick/fast).
        # TTS is intentionally NOT overridden here — its progress_callback
        # already updated the loading screen during warmup, and unconditionally
        # setting "ready" would mask warmup failures.
        loading.update_provider("stt", "ready", 1.0)
        loading.update_provider("pi", "ready", 1.0)

        # Remove loading screen — warmup completed or failed non-fatally
        try:
            await self.pop_screen()
        except Exception:
            pass  # screen may have already been removed

        try:
            await pipeline.run()
        except asyncio.CancelledError:
            pass  # normal shutdown via app exit
