"""RPC-based agent provider — persistent Pi subprocess with conversation memory.

The subprocess communicates via LF-delimited JSONL over stdin/stdout and is
kept alive across turns.  Conversation history is managed by Pi itself via
``--session-dir``.

Pi streams content blocks, tool calls, and completion events on **stdout**
in real time.  We read the full JSONL stream directly — no session-file
polling.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
from collections.abc import AsyncIterator

# StreamReader default limit for subprocess stdout (bytes).
# Default is 65536 (64 KiB). Pi may emit JSONL lines larger than that
# (e.g. large tool results, code blocks) which would raise
# LimitOverrunError: "Separator is found, but chunk is longer than limit".
_STDOUT_BUFFER_LIMIT = 2 ** 22  # 4 MiB

from whisper_bot.agent import AgentProvider
from whisper_bot.debug import log as _debug
from whisper_bot.debug import log_exc as _debug_exc
from whisper_bot.events import (
    EventBus,
    PipelineError,
    ToolExecutionEnd,
    ToolExecutionStart,
)


class PiRpcAgentProvider(AgentProvider):
    """Persistent Pi subprocess with conversation memory.

    Pi runs as a persistent ``pi --mode rpc`` subprocess.  Prompts are sent
    via stdin JSONL; the model response is streamed back as JSONL on stdout
    (content blocks, tool calls, completion events).

    Config keys read from ``config["agent"]``:

    * ``timeout_seconds`` — per-read timeout (default 120).
    * ``session_dir`` — directory for persistent sessions
      (default ``~/.whisper-bot/sessions``).
    * ``resume_session`` — whether to persist history across restarts
      (default ``False``).
    """

    def __init__(self, config: dict, event_bus: EventBus) -> None:
        self._event_bus = event_bus
        self._timeout = config.get("agent", {}).get("timeout_seconds", 120)
        self._lock = asyncio.Lock()
        self._req_id = 0
        self._shutdown_requested = False

        # Session management
        agent_cfg = config.get("agent", {})
        self._resume_session = agent_cfg.get("resume_session", False)
        session_dir = agent_cfg.get("session_dir", "~/.whisper-bot/sessions")
        self._session_dir = os.path.expanduser(session_dir)

        # Auto-restart backoff (seconds; 0 = no delay)
        self._backoff = 0

        self._process: asyncio.subprocess.Process | None = None

        _debug(
            f"[DEBUG pi_rpc] PiRpcAgentProvider initialized: "
            f"timeout={self._timeout}s "
            f"session={'enabled' if self._resume_session else 'disabled'}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def warmup(self) -> None:
        """Eagerly start the persistent Pi subprocess at pipeline init.

        This ensures ``pi --mode rpc`` is running and ready before the
        first voice turn arrives, so the user doesn't wait for subprocess
        spawn + model load when they start speaking.
        """
        _debug("[DEBUG pi_rpc] Warmup: starting persistent subprocess...")
        try:
            await self._ensure_alive()
            _debug("[DEBUG pi_rpc] Warmup complete")
        except Exception as exc:
            _debug(f"[DEBUG pi_rpc] Warmup failed (non-fatal): {exc}")

    async def process(self, text: str) -> AsyncIterator[str]:
        """Process *text* via the persistent RPC subprocess.

        Sends a prompt to Pi via stdin, then reads the **full stdout JSONL
        stream** — Pi sends content blocks, tool calls, and completion
        events on stdout in real time.  No session-file polling needed.
        """
        if self._shutdown_requested:
            _debug("[DEBUG pi_rpc] Shutdown requested — refusing new process() call")
            return

        async with self._lock:
            _debug(f"[DEBUG pi_rpc] process() called with text ({len(text)} chars)")
            _debug(f"[DEBUG pi_rpc] Text starts with: {text[:200]!r}")

            # 1. Ensure subprocess is alive (re-spawn if dead)
            try:
                await self._ensure_alive()
            except Exception:
                _debug("[DEBUG pi_rpc] Could not ensure subprocess — aborting turn")
                return

            # 2. Build request id and send prompt
            req_id = f"req_{self._req_id}"
            self._req_id += 1
            request = {"id": req_id, "type": "prompt", "message": text}

            await self._send_jsonl(request)

            # 3. Read the full stdout stream — Pi sends content blocks,
            #    tool calls, and completion events on stdout in real time.
            _debug(f"[DEBUG pi_rpc] Reading stdout stream for req_id={req_id}...")
            full_text = ""
            async for token in self._read_stdout_stream(req_id):
                full_text += token
                yield token

            _debug(f"[DEBUG pi_rpc] Turn complete: full_text={len(full_text)} chars")

            # Reset backoff on a successful turn
            self._backoff = 0

    async def shutdown(self) -> None:
        """Cleanly shut down the persistent subprocess."""
        self._shutdown_requested = True
        _debug("[DEBUG pi_rpc] shutdown() called")

        proc = self._process
        if proc is not None and proc.returncode is None:
            pid = proc.pid
            _debug(f"[DEBUG pi_rpc] Sending SIGTERM to pid {pid}")
            try:
                proc.send_signal(signal.SIGTERM)
                await asyncio.wait_for(proc.wait(), timeout=5.0)
                _debug("[DEBUG pi_rpc] Subprocess terminated gracefully")
            except asyncio.TimeoutError:
                _debug("[DEBUG pi_rpc] SIGTERM timed out — sending SIGKILL")
                try:
                    proc.send_signal(signal.SIGKILL)
                    await proc.wait()
                    _debug("[DEBUG pi_rpc] Subprocess killed")
                except ProcessLookupError:
                    _debug("[DEBUG pi_rpc] Process already dead after SIGKILL")
            except ProcessLookupError:
                _debug("[DEBUG pi_rpc] Process already dead before SIGTERM")

        self._process = None

    # ------------------------------------------------------------------
    # Subprocess lifecycle
    # ------------------------------------------------------------------

    async def _ensure_alive(self) -> None:
        """Spawn or re-spawn the subprocess if needed.

        Applies exponential backoff (1 s, 2 s, 4 s, … max 30 s) before
        retrying after an unexpected death.  Raises on failure after
        emitting a ``PipelineError`` on the bus.
        """
        if self._process is not None and self._process.returncode is None:
            return  # already alive

        # Apply backoff delay before retry
        if self._backoff > 0:
            _debug(f"[DEBUG pi_rpc] Backoff: sleeping {self._backoff}s before restart")
            await asyncio.sleep(self._backoff)

        # Increase backoff for the *next* failure (before attempting spawn)
        self._backoff = min(max(self._backoff * 2, 1), 30) if self._backoff else 1

        await self._spawn()

        # Spawn succeeded — reset backoff
        self._backoff = 0

    def _clean_old_sessions(self) -> None:
        """Remove session files so Pi starts a fresh conversation."""
        if not os.path.isdir(self._session_dir):
            return
        for fname in os.listdir(self._session_dir):
            if fname.endswith((".jsonl", ".json", ".db", ".sqlite")):
                fpath = os.path.join(self._session_dir, fname)
                try:
                    if os.path.isfile(fpath):
                        os.unlink(fpath)
                        _debug(f"[DEBUG pi_rpc] Removed stale session file: {fname}")
                except OSError as exc:
                    _debug(f"[DEBUG pi_rpc] Failed to remove {fname}: {exc}")

    async def _spawn(self) -> None:
        """Spawn the ``pi --mode rpc`` subprocess.

        Always uses ``--session-dir`` — Pi requires a session to process
        prompts with conversation memory.
        """
        os.makedirs(self._session_dir, exist_ok=True)
        args = ["pi", "--mode", "rpc", "--session-dir", self._session_dir]
        self._clean_old_sessions()

        _debug(f"[DEBUG pi_rpc] Spawning: {' '.join(args)}")

        try:
            self._process = await asyncio.create_subprocess_exec(
                *args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            # Bump stdout buffer limit — Pi may emit lines larger than the
            # default 64 KiB (tool results, code blocks).  Without this,
            # readline() raises LimitOverrunError.
            if self._process.stdout is not None:
                self._process.stdout._limit = _STDOUT_BUFFER_LIMIT
            _debug(f"[DEBUG pi_rpc] Subprocess spawned with pid={self._process.pid}")
        except FileNotFoundError:
            _debug("[DEBUG pi_rpc] FAILED: 'pi' binary not found on PATH!")
            await self._event_bus.emit(
                PipelineError(
                    stage="agent",
                    fatal=True,
                    message="Pi agent CLI not found. Is it installed?",
                )
            )
            raise
        except Exception as exc:
            _debug(f"[DEBUG pi_rpc] FAILED to spawn subprocess: {type(exc).__name__}: {exc}")
            _debug_exc()
            await self._event_bus.emit(
                PipelineError(
                    stage="agent",
                    fatal=True,
                    message=f"Failed to spawn Pi agent: {exc}",
                )
            )
            raise

    # ------------------------------------------------------------------
    # RPC I/O — stdin write, stdout stream read
    # ------------------------------------------------------------------

    async def _send_jsonl(self, data: dict) -> None:
        """Serialize *data* as JSON and write a LF-delimited line to stdin."""
        assert self._process is not None
        assert self._process.stdin is not None

        line = json.dumps(data, ensure_ascii=False) + "\n"
        _debug(f"[DEBUG pi_rpc] SEND: {line.strip()}")
        self._process.stdin.write(line.encode("utf-8"))
        await self._process.stdin.drain()

    # ------------------------------------------------------------------
    # Stdout stream reader — the core of RPC communication
    # ------------------------------------------------------------------

    async def _read_stdout_stream(self, req_id: str) -> AsyncIterator[str]:
        """Read the full stdout JSONL stream from Pi after sending a prompt.

        Pi's RPC protocol sends messages on stdout in this order:

        1. Ack (``{"id":"req_N","type":"response","command":"prompt","success":true}``).
        2. ``agent_start`` — generation started.
        3. ``turn_start`` — a new agent turn begins.
        4. **Streaming content** — delivered as ``message_update`` events:
           - ``text_delta`` — incremental text fragments in ``assistantMessageEvent.delta``
           - ``toolcall_*`` — tool calls within the message
        5. ``turn_end`` — turn completes.
        6. ``agent_end`` — generation complete (terminal; the method returns).

        The actual text content is in ``message_update`` → ``assistantMessageEvent``
        → ``{"type":"text_delta","delta":"hello"}``, NOT in the old session-file
        format that ``_read_ack`` + ``_poll_session_file`` previously used.

        This method reads the **entire** stream, yielding text from ``text_delta``
        events and emitting tool execution events on the bus as they arrive.
        Returns on ``agent_end``, EOF, or deadline expiry.
        """
        assert self._process is not None
        assert self._process.stdout is not None

        deadline = asyncio.get_event_loop().time() + self._timeout
        PER_READ_TIMEOUT = 5.0  # seconds — short so expiry falls through to deadline
        got_ack = False

        while asyncio.get_event_loop().time() < deadline:
            try:
                line_bytes = await asyncio.wait_for(
                    self._process.stdout.readline(),
                    timeout=PER_READ_TIMEOUT,
                )
            except asyncio.TimeoutError:
                # Per-read timeout reached but overall deadline may still be open
                if asyncio.get_event_loop().time() < deadline:
                    continue  # keep reading
                _debug("[DEBUG pi_rpc] Timeout reading stdout")
                await self._event_bus.emit(
                    PipelineError(
                        stage="agent", fatal=False,
                        message="Pi did not respond in time",
                    )
                )
                return
            except (asyncio.LimitOverrunError, ValueError) as exc:
                # Line exceeded buffer limit (should not happen after the
                # _limit bump in _spawn, but guard against edge cases).
                _debug(f"[DEBUG pi_rpc] Buffer limit exceeded: {exc}")
                # Drain past the newline in the internal buffer so the
                # stream reader can continue on the next line.
                try:
                    buf = self._process.stdout._buffer
                    idx = buf.find(b"\n")
                    if idx >= 0:
                        self._process.stdout._buffer = buf[idx + 1:]
                    else:
                        self._process.stdout._buffer = b""
                except Exception:
                    self._process.stdout._buffer = b""
                continue

            if not line_bytes:
                _debug("[DEBUG pi_rpc] EOF on stdout")
                return

            decoded = line_bytes.decode(errors="replace").strip()
            _debug(f"[DEBUG pi_rpc] STDOUT: {decoded[:300]}")

            try:
                msg = json.loads(decoded)
            except json.JSONDecodeError:
                _debug(f"[DEBUG pi_rpc] Non-JSON stdout: {decoded[:200]}")
                continue

            msg_type = msg.get("type", "")

            # ---- Before ack: only care about agent_start and the ack ----
            if not got_ack:
                if msg_type == "agent_start":
                    _debug("[DEBUG pi_rpc] Agent started")
                    continue

                if msg.get("id") == req_id:
                    if msg.get("success"):
                        _debug("[DEBUG pi_rpc] Ack received")
                        got_ack = True
                        continue
                    else:
                        err = msg.get("error", "Unknown error")
                        _debug(f"[DEBUG pi_rpc] Pi returned error: {err}")
                        await self._event_bus.emit(
                            PipelineError(
                                stage="agent", fatal=False, message=err,
                            )
                        )
                        return

                # Any other pre-ack message — skip (stale events, etc.)
                continue

            # ---- After ack: content, tool calls, completion ----

            # Terminal event — agent finished the full turn
            if msg_type == "agent_end":
                _debug("[DEBUG pi_rpc] Agent end — turn complete")
                return

            # ------------------------------------------------------------------
            # Pi RPC streaming format: message_update with an
            # assistantMessageEvent sub-object that carries the delta.
            # ------------------------------------------------------------------
            if msg_type == "message_update":
                event = msg.get("assistantMessageEvent", {})
                event_type = event.get("type", "")

                # Text content (the actual response text)
                if event_type == "text_delta":
                    text = event.get("delta", "")
                    if text.strip():
                        yield text

                # Text block lifecycle — markers, no content
                elif event_type in ("text_start", "text_end"):
                    pass

                # Tool calls inside the assistant message
                elif event_type == "toolcall_start":
                    partial = event.get("partial", {})
                    content = partial.get("content", [])
                    # Find the tool call block at contentIndex
                    idx = event.get("contentIndex", -1)
                    if idx >= 0 and idx < len(content):
                        block = content[idx]
                        if isinstance(block, dict) and block.get("type") == "toolUse":
                            await self._event_bus.emit(
                                ToolExecutionStart(
                                    tool_call_id=block.get("id", ""),
                                    tool_name=block.get("name", ""),
                                    args=json.dumps(
                                        block.get("input", {}),
                                        ensure_ascii=False,
                                    ),
                                )
                            )
                elif event_type == "toolcall_end":
                    tool_call = event.get("toolCall", {})
                    if tool_call:
                        await self._event_bus.emit(
                            ToolExecutionEnd(
                                tool_call_id=tool_call.get("id", ""),
                                tool_name=tool_call.get("name", "") or "tool",
                                result=json.dumps(
                                    tool_call.get("input", {}),
                                    ensure_ascii=False,
                                ),
                            )
                        )

                # Thinking blocks — skip (not user-facing text)
                elif event_type in ("thinking_start", "thinking_delta", "thinking_end"):
                    pass

                # Message-complete marker — not terminal for the turn
                elif event_type == "done":
                    _debug("[DEBUG pi_rpc] Message done (stream continues)")

                # Error in the stream
                elif event_type == "error":
                    reason = event.get("reason", "unknown")
                    _debug(f"[DEBUG pi_rpc] Stream error: {reason}")
                    await self._event_bus.emit(
                        PipelineError(
                            stage="agent", fatal=False,
                            message=f"Pi stream error: {reason}",
                        )
                    )

                continue  # handled; don't fall through to fallback handlers

            # ------------------------------------------------------------------
            # Pi RPC tool execution events (flat, outside message_update)
            # ------------------------------------------------------------------
            if msg_type == "tool_execution_start":
                await self._event_bus.emit(
                    ToolExecutionStart(
                        tool_call_id=msg.get("toolCallId", ""),
                        tool_name=msg.get("toolName", ""),
                        args=json.dumps(
                            msg.get("args", {}), ensure_ascii=False,
                        ),
                    )
                )
                continue

            if msg_type == "tool_execution_end":
                result = json.dumps(
                    msg.get("result", {}), ensure_ascii=False,
                )
                if len(result) > 2000:
                    result = result[:2000] + "... [truncated]"
                await self._event_bus.emit(
                    ToolExecutionEnd(
                        tool_call_id=msg.get("toolCallId", ""),
                        tool_name=msg.get("toolName", "") or "tool",
                        result=result,
                    )
                )
                continue

            # tool_execution_update — accumulative partial output, skip for now
            if msg_type == "tool_execution_update":
                continue

            # ------------------------------------------------------------------
            # Fallback handlers (session-file format / legacy formats)
            # ------------------------------------------------------------------

            # Content in session-file format:
            #   {"type":"message","message":{"role":"assistant",
            #     "content":[{"type":"text","text":"..."}]}}
            if msg_type == "message":
                inner = msg.get("message", {})
                role = inner.get("role")
                content = inner.get("content", [])

                if role == "assistant":
                    for block in content:
                        if not isinstance(block, dict):
                            continue
                        if block.get("type") == "text":
                            text = block.get("text", "")
                            if text.strip():
                                yield text
                        elif block.get("type") == "toolCall":
                            await self._event_bus.emit(
                                ToolExecutionStart(
                                    tool_call_id=block.get("id", ""),
                                    tool_name=block.get("name", ""),
                                    args=json.dumps(
                                        block.get("arguments", {}),
                                        ensure_ascii=False,
                                    ),
                                )
                            )
                elif role == "toolResult":
                    tc_id = inner.get("toolCallId", "")
                    tc_name = inner.get("toolName", "")
                    result = json.dumps(
                        inner.get("details", {}), ensure_ascii=False,
                    )
                    if len(result) > 2000:
                        result = result[:2000] + "... [truncated]"
                    await self._event_bus.emit(
                        ToolExecutionEnd(
                            tool_call_id=tc_id,
                            tool_name=tc_name or "tool",
                            result=result,
                        )
                    )

            # Direct text content — alternative format Pi might use
            elif msg_type in ("text", "content_block"):
                text = msg.get("text", "") or msg.get("content", "")
                if text.strip():
                    yield text

            # Direct tool events — alternative format
            elif msg_type in ("toolCall", "tool_call"):
                await self._event_bus.emit(
                    ToolExecutionStart(
                        tool_call_id=msg.get("id", ""),
                        tool_name=msg.get("name", ""),
                        args=json.dumps(
                            msg.get("arguments", msg.get("args", {})),
                            ensure_ascii=False,
                        ),
                    )
                )
            elif msg_type in ("toolResult", "tool_result"):
                result = json.dumps(
                    msg.get("details", msg.get("result", {})),
                    ensure_ascii=False,
                )
                if len(result) > 2000:
                    result = result[:2000] + "... [truncated]"
                await self._event_bus.emit(
                    ToolExecutionEnd(
                        tool_call_id=msg.get("toolCallId", msg.get("id", "")),
                        tool_name=msg.get("toolName", msg.get("name", "tool")),
                        result=result,
                    )
                )

        # Overall deadline expired
        _debug(f"[DEBUG pi_rpc] Deadline ({self._timeout}s) reached")
        await self._event_bus.emit(
            PipelineError(
                stage="agent", fatal=False,
                message="Pi did not produce a complete response in time",
            )
        )
