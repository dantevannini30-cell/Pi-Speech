"""Terminal display for the whisper-bot pipeline.

Uses a simple chat-thread layout — no ANSI cursor manipulation, which avoids
garbled output from mixed stdout/stderr and stdin echo.

Layout::

    whisper-bot> listen
      [recording started — type 'stop' to end]
    whisper-bot> stop
      [transcribing...]
      You: This is a test.
    Pi: Sure — I'll help you with that.
      [done]
    whisper-bot>

All status messages go to *stderr*; the agent's streaming tokens go to *stdout*
so they can be piped separately.
"""

from __future__ import annotations
from whisper_bot.debug import log as _debug

import shutil
import sys

# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


class Display:
    """Renders pipeline state changes to the terminal."""

    def __init__(self) -> None:
        self._column: int = 0
        _debug("[DEBUG display] Display created")

    # -- Status (stderr) --------------------------------------------------------

    def show_status(self, text: str) -> None:
        """Print a status line to stderr (e.g. recording, transcribing, error)."""
        _debug(f"[DEBUG display] show_status: {text!r}")
        print(text, file=sys.stderr, flush=True)

    # -- Line output (stdout) ---------------------------------------------------

    def replace_line(self, text: str) -> None:
        """Write a full line to stdout with a newline."""
        _debug(f"[DEBUG display] replace_line: {text!r}")
        print(text, flush=True)

    # -- Token streaming (stdout) -----------------------------------------------

    def stream_token(self, token: str) -> None:
        """Stream *token* to stdout, wrapping at the terminal width."""
        original = token
        token = token.replace("\n", "")

        width = shutil.get_terminal_size().columns
        if self._column + len(token) > width:
            sys.stdout.write("\n")
            self._column = 0

        sys.stdout.write(token)
        self._column += len(token)
        sys.stdout.flush()
        if len(token) < len(original):
            _debug(f"[DEBUG display] stream_token (stripped {len(original) - len(token)} newlines): {token!r}")
        elif self._column < 80:
            _debug(f"[DEBUG display] stream_token: {token!r}")

    # -- Newline ----------------------------------------------------------------

    def newline(self) -> None:
        """Write a newline and reset the column tracker."""
        _debug("[DEBUG display] newline()")
        sys.stdout.write("\n")
        sys.stdout.flush()
        self._column = 0

    # -- Prompt ----------------------------------------------------------------

    def show_prompt(self) -> None:
        """Print 'whisper-bot> ' without a trailing newline."""
        _debug("[DEBUG display] show_prompt()")
        sys.stdout.write("whisper-bot> ")
        sys.stdout.flush()
        self._column = 0

    # -- Line clearing ---------------------------------------------------------

    def clear_current_line(self) -> None:
        """Clear the current line and flush."""
        _debug("[DEBUG display] clear_current_line()")
        sys.stdout.write("\033[K")
        sys.stdout.flush()

    # -- Cursor ----------------------------------------------------------------

    def hide_cursor(self) -> None:
        """Write the hide-cursor escape sequence."""
        _debug("[DEBUG display] hide_cursor()")
        sys.stdout.write("\033[?25l")
        sys.stdout.flush()

    def show_cursor(self) -> None:
        """Write the show-cursor escape sequence."""
        _debug("[DEBUG display] show_cursor()")
        sys.stdout.write("\033[?25h")
        sys.stdout.flush()

    # -- Agent label ------------------------------------------------------------

    def show_agent_label(self) -> None:
        """Write the agent response label (``  Pi: ``) to stdout.

        Called by ``Pipeline._run_agent`` before the first token streams.
        """
        _debug("[DEBUG display] show_agent_label()")
        sys.stdout.write("  Pi: ")
        sys.stdout.flush()
        self._column = 0