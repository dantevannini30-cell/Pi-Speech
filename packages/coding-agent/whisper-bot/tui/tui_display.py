"""Drop-in Display replacement that routes pipeline output to TUI widgets."""

from __future__ import annotations

from whisper_bot.tui.chat_pane import ChatPane
from whisper_bot.tui.status_bar import WhisperStatusBar


class TuiDisplay:
    """Compatible with ``Display`` — routes all output to TUI widgets.

    Does not inherit from ``Display``; only needs the same method signatures
    that ``Pipeline`` calls on ``self.display``.
    """

    def __init__(
        self,
        chat_pane: ChatPane,
        status_bar: WhisperStatusBar,
    ) -> None:
        self._chat = chat_pane
        self._status = status_bar

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    def show_status(self, text: str) -> None:
        """Route a status message to the chat pane."""
        text = text.strip().strip("[]").strip()
        if text:
            self._chat.add_status(text)

    # ------------------------------------------------------------------
    # Line output
    # ------------------------------------------------------------------

    def replace_line(self, text: str) -> None:
        """Route a full line to the chat pane as a user or agent message."""
        text = text.strip()
        if text.startswith("You:"):
            self._chat.add_user_message(text[4:].strip())
        elif text.startswith("Pi:"):
            self._chat.start_agent_message()

    # ------------------------------------------------------------------
    # Token streaming
    # ------------------------------------------------------------------

    def stream_token(self, token: str) -> None:
        """Stream a token to the current agent message."""
        self._chat.stream_token(token)

    # ------------------------------------------------------------------
    # Agent label
    # ------------------------------------------------------------------

    def show_agent_label(self) -> None:
        """Show the agent response label (start a new agent message)."""
        self._chat.start_agent_message()

    # ------------------------------------------------------------------
    # Newline
    # ------------------------------------------------------------------

    def newline(self) -> None:
        """Finalize the current agent message."""
        self._chat.finalize_message()

    # ------------------------------------------------------------------
    # Prompt (no-op in TUI mode)
    # ------------------------------------------------------------------

    def show_prompt(self) -> None:
        """No prompt shown in TUI mode."""

    # ------------------------------------------------------------------
    # Line clearing (no-op in TUI mode)
    # ------------------------------------------------------------------

    def clear_current_line(self) -> None:
        """Not needed in TUI mode."""

    # ------------------------------------------------------------------
    # Cursor (no-op in TUI mode)
    # ------------------------------------------------------------------

    def hide_cursor(self) -> None:
        """Not needed in TUI mode."""

    def show_cursor(self) -> None:
        """Not needed in TUI mode."""
