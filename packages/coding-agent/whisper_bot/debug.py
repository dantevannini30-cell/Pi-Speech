"""Simple debug-print helper controlled by a global --debug flag.

In TUI mode debug output is redirected to a file so it doesn't get lost
in terminal rendering.  Tail ``~/.whisper-bot/debug.log`` to follow along.
"""

from __future__ import annotations

import os
import sys

_enabled = False
_file: str | None = None
_fh: object | None = None  # open file handle, typed as object to avoid pyright


def set_enabled(enabled: bool) -> None:
    global _enabled
    _enabled = enabled


def set_file(path: str) -> None:
    """Redirect debug output to *path* instead of stderr."""
    global _file, _fh
    _file = path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _fh = open(path, "a", buffering=1)  # line-buffered


def _out() -> object:
    """Return the current output stream (stderr or file)."""
    return _fh if _fh is not None else sys.stderr


def log(msg: str) -> None:
    """Print *msg* to debug output with flush, but only if debug is enabled."""
    if _enabled:
        out = _out()
        print(msg, file=out, flush=True)


def log_exc() -> None:
    """Print the current exception traceback to debug output, but only if debug is enabled."""
    if _enabled:
        import traceback
        traceback.print_exc(file=_out())
