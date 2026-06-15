"""Agent provider that spawns the `pi` CLI subprocess for each request."""

from __future__ import annotations
from whisper_bot.debug import log as _debug
from whisper_bot.debug import log_exc as _debug_exc

import asyncio
import os
import signal
import subprocess
import sys
from collections.abc import AsyncIterator

from whisper_bot.agent import AgentProvider
from whisper_bot.events import EventBus, PipelineError


class PiAgentProvider(AgentProvider):
    """Spawns ``pi -p <text>`` and streams stdout line-by-line as tokens."""

    def __init__(self, config: dict, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        working_dir = config.get("agent", {}).get("working_directory", "~")
        self._working_directory = os.path.expanduser(working_dir)
        self._timeout = config.get("agent", {}).get("timeout_seconds", 120)
        self._process: asyncio.subprocess.Process | None = None
        _debug(f"[DEBUG agent] PiAgentProvider initialized: working_dir={self._working_directory} timeout={self._timeout}s")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def process(self, text: str) -> AsyncIterator[str]:
        """Process *text* by spawning ``pi -p <text>``.

        Yields each line from stdout as a token.  On completion emits
        ``AgentDone``.  On any failure emits ``PipelineError(stage="agent",
        fatal=True)``.
        """
        self._process = None
        full_text = ""

        _debug(f"[DEBUG agent] process() called with text ({len(text)} chars)")
        _debug(f"[DEBUG agent] Text starts with: {text[:200]!r}")

        # ----- spawn -----------------------------------------------------------
        try:
            _debug(f"[DEBUG agent] Spawning: pi -p <text>  (cwd={self._working_directory})")
            self._process = await asyncio.create_subprocess_exec(
                "pi",
                "-p",
                text,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self._working_directory,
            )
            _debug(f"[DEBUG agent] Subprocess spawned with pid={self._process.pid}")
        except FileNotFoundError as exc:
            _debug(f"[DEBUG agent] FAILED: 'pi' binary not found on PATH!")
            await self._event_bus.emit(
                PipelineError(
                    stage="agent",
                    fatal=True,
                    message=f"Pi agent CLI not found: {exc}. Is it installed?",
                )
            )
            return
        except Exception as exc:
            _debug(f"[DEBUG agent] FAILED to spawn subprocess: {type(exc).__name__}: {exc}")
            await self._event_bus.emit(
                PipelineError(
                    stage="agent",
                    fatal=True,
                    message=f"Failed to spawn Pi agent: {exc}",
                )
            )
            return

        # ----- consume stdout --------------------------------------------------
        try:
            line_count = 0
            async for token in self._read_tokens():
                line_count += 1
                full_text += token
                _debug(f"[DEBUG agent] Token #{line_count} ({len(token)} chars): {token[:100]!r}")
                yield token

            _debug(f"[DEBUG agent] EOF on stdout, read {line_count} lines total")

            # ----- check exit code ---------------------------------------------
            _debug(f"[DEBUG agent] Waiting for subprocess to exit (timeout={self._timeout}s)...")
            return_code = await asyncio.wait_for(
                self._process.wait(),
                timeout=self._timeout,
            )
            _debug(f"[DEBUG agent] Subprocess exited with code {return_code}")

            if return_code != 0:
                stderr_data = await self._process.stderr.read()
                stderr_text = stderr_data.decode(errors="replace").strip()
                _debug(f"[DEBUG agent] stderr output: {stderr_text[:500]!r}")
                await self._event_bus.emit(
                    PipelineError(
                        stage="agent",
                        fatal=True,
                        message=(
                            f"Pi agent exited with code {return_code}"
                            f"{': ' + stderr_text if stderr_text else ''}"
                        ),
                    )
                )
                return

            _debug(f"[DEBUG agent] Agent succeeded, full_text ({len(full_text)} chars): {full_text[:200]!r}")

        except asyncio.TimeoutError:
            _debug(f"[DEBUG agent] TIMEOUT: no output for {self._timeout}s, sending SIGTERM")
            self._signal_timeout()
            await self._event_bus.emit(
                PipelineError(
                    stage="agent",
                    fatal=True,
                    message="Pi agent timed out",
                )
            )
        except Exception as exc:
            _debug(f"[DEBUG agent] UNEXPECTED exception: {type(exc).__name__}: {exc}")
            _debug_exc()
            await self._event_bus.emit(
                PipelineError(
                    stage="agent",
                    fatal=True,
                    message=str(exc),
                )
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _read_tokens(self) -> AsyncIterator[str]:
        """Read stdout one line at a time with per-read timeout."""
        assert self._process is not None
        assert self._process.stdout is not None

        while True:
            try:
                line_bytes = await asyncio.wait_for(
                    self._process.stdout.readline(),
                    timeout=self._timeout,
                )
            except asyncio.TimeoutError:
                _debug(f"[DEBUG agent] readline() timed out after {self._timeout}s")
                raise  # let the outer handler send SIGTERM

            if not line_bytes:  # EOF
                _debug("[DEBUG agent] readline() returned empty bytes = EOF")
                break

            decoded = line_bytes.decode(errors="replace").rstrip("\n")
            _debug(f"[DEBUG agent] readline() raw ({len(line_bytes)} bytes): {line_bytes[:100]!r}")
            yield decoded

    def _signal_timeout(self) -> None:
        """Send SIGTERM to the running subprocess if it is still alive."""
        if self._process is not None and self._process.returncode is None:
            try:
                _debug(f"[DEBUG agent] Sending SIGTERM to pid {self._process.pid}")
                self._process.send_signal(signal.SIGTERM)
            except ProcessLookupError:
                _debug(f"[DEBUG agent] Process {self._process.pid} already dead")
                pass  # already dead — nothing to signal