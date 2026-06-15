"""Async stdin input handler for whisper-bot.

Reads commands from stdin and publishes them as events on the bus.
"""

from __future__ import annotations
from whisper_bot.debug import log as _debug

import asyncio
import fcntl
import os
import select
import sys
import termios
import tty

from whisper_bot.events import EventBus
from whisper_bot.events.types import InputCommand, PipelineError


class InputHandler:
    """Reads lines from stdin and emits typed events onto the event bus.

    Parameters
    ----------
    event_bus:
        Shared event bus for emitting user commands.
    shutdown_event:
        Fired to signal all tasks to stop.
    pipeline_idle:
        Tracks whether the pipeline is idle.  The InputHandler waits
        on this event before reading the next line, and clears it
        after emitting ``stop`` so no new stdin is consumed while
        TypeWhisper transcribes and types text into the terminal.
    """

    def __init__(
        self,
        event_bus: EventBus,
        shutdown_event: asyncio.Event | None = None,
        pipeline_idle: asyncio.Event | None = None,
    ) -> None:
        self._bus = event_bus
        self._shutdown = shutdown_event or asyncio.Event()
        self._pipeline_idle = pipeline_idle or asyncio.Event()
        self._pipeline_idle.set()  # start idle
        self._cancelled: bool = False
        _debug("[DEBUG input] InputHandler created")

    async def run(self) -> None:
        """Loop reading stdin until cancelled."""
        _debug("[DEBUG input] InputHandler.run() started, waiting for stdin...")
        loop = asyncio.get_event_loop()
        while not self._cancelled and not self._shutdown.is_set():
            # Don't read stdin while the pipeline is busy (TypeWhisper
            # may be typing transcribed text into our terminal).
            _debug("[DEBUG input] Waiting for pipeline idle...")
            await self._pipeline_idle.wait()

            # Drain any text TypeWhisper typed into the terminal while
            # the pipeline was processing (before showing the prompt).
            # Wait a beat first so TypeWhisper has time to finish typing.
            await asyncio.sleep(0.2)
            _debug("[DEBUG input] Pipeline idle — draining stdin if needed")
            drained = await loop.run_in_executor(None, self._drain_stdin)
            if drained:
                _debug(f"[DEBUG input] Drained {drained} bytes from stdin (TypeWhisper-typed text)")

            # Show the prompt so the user knows they can type.
            sys.stdout.write("whisper-bot> ")
            sys.stdout.flush()

            # run_in_executor won't truly cancel, so we also check the
            # event before *and* after each readline.
            _debug("[DEBUG input] Calling readline (blocking in executor)...")
            line = await loop.run_in_executor(None, sys.stdin.readline)

            if self._cancelled or self._shutdown.is_set():
                _debug("[DEBUG input] Cancelled or shutdown after readline, breaking")
                break

            cmd = line.strip()
            _debug(f"[DEBUG input] Read line: {line!r} -> cmd={cmd!r}")

            if not cmd:
                _debug("[DEBUG input] Empty line, continuing")
                continue

            if cmd == "listen":
                _debug("[DEBUG input] Emitting InputCommand(listen)")
                await self._bus.emit(InputCommand(action="listen"))
            elif cmd == "stop":
                _debug("[DEBUG input] Emitting InputCommand(stop)")
                await self._bus.emit(InputCommand(action="stop"))
                # Don't read any more stdin until the pipeline finishes
                # its current cycle (TypeWhisper will type text).
                self._pipeline_idle.clear()
            elif cmd == "quit":
                _debug("[DEBUG input] Emitting InputCommand(quit) and shutting down")
                await self._bus.emit(InputCommand(action="quit"))
                self._cancelled = True
                self._shutdown.set()
            elif cmd == "help":
                _debug("[DEBUG input] Emitting InputCommand(help)")
                await self._bus.emit(InputCommand(action="help"))
            else:
                _debug(f"[DEBUG input] Unknown command: {cmd!r}, emitting PipelineError")
                await self._bus.emit(
                    PipelineError(
                        stage="input",
                        message=f"Unknown command: {cmd}",
                        fatal=False,
                    )
                )

        _debug("[DEBUG input] InputHandler.run() exiting")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _drain_stdin() -> int:
        """Drain stdin of any pending characters, even without a trailing newline.

        TypeWhisper types transcribed text into the terminal via accessibility
        APIs, but the text lands in the kernel's line-discipline buffer (the
        terminal is in canonical mode).  In canonical mode ``read()`` will not
        return data until a newline arrives, so we temporarily switch to
        non-canonical mode (VMIN=0, VTIME=0) to pull whatever TypeWhisper
        typed in, then restore the original settings.

        If stdin is a pipe (not a TTY), we use the simpler O_NONBLOCK path.
        """
        total = 0
        fd = sys.stdin.fileno()

        # --- TTY path — temporarily switch to non-canonical mode ---
        # This works on a real terminal where TypeWhisper types characters
        # without a trailing newline.  On a pipe termios calls raise
        # termios.error / OSError and we fall through to the pipe path.
        old_tty = None
        try:
            old_tty = termios.tcgetattr(fd)
            new = termios.tcgetattr(fd)
            # Non-canonical, no echo, VMIN=0 VTIME=0 (immediate no-wait read)
            new[tty.LFLAG] &= ~(termios.ICANON | termios.ECHO)
            new[tty.CC][termios.VMIN] = 0
            new[tty.CC][termios.VTIME] = 0
            termios.tcsetattr(fd, termios.TCSANOW, new)

            while True:
                data = os.read(fd, 4096)
                if not data:
                    break
                total += len(data)
            return total  # TTY — all characters consumed
        except (OSError, termios.error):
            pass  # not a TTY — fall through to pipe path below
        finally:
            if old_tty is not None:
                try:
                    termios.tcsetattr(fd, termios.TCSANOW, old_tty)
                except (OSError, termios.error):
                    pass

        # --- Pipe / redirected-stdin path ---
        # O_NONBLOCK + select.select works correctly here (no canonical
        # buffering, so every byte written to the pipe is immediately
        # visible to select).
        try:
            fl = fcntl.fcntl(fd, fcntl.F_GETFL)
            fcntl.fcntl(fd, fcntl.F_SETFL, fl | os.O_NONBLOCK)
            while select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                data = os.read(fd, 4096)
                if not data:
                    break
                total += len(data)
        except (BlockingIOError, OSError):
            pass  # no more data
        finally:
            try:
                fcntl.fcntl(fd, fcntl.F_SETFL, fl)
            except OSError:
                pass

        return total

    def cancel(self) -> None:
        """Signal the run loop to exit."""
        _debug("[DEBUG input] cancel() called")
        self._cancelled = True