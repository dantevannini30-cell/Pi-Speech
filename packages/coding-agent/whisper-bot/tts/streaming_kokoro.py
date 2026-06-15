"""Streaming Kokoro TTS — speaks sentences as they arrive.

Uses a **pool** of persistent Python subprocesses that pre-import torch /
numpy / KPipeline once at startup.  Sentences are dispatched to available
workers via a two-phase JSONL protocol:

    1. ``speak`` → worker generates audio and replies ``ready``
    2. ``play``  → worker plays the pre-generated audio and replies ``done``

This lets sentence N+1 generate while sentence N plays, hiding generation
latency behind playback time.
"""

from __future__ import annotations

import asyncio
import collections.abc
import json
import os
import signal
import subprocess
import sys
from typing import Any, Awaitable

from whisper_bot.debug import log as _debug
from whisper_bot.debug import _enabled as _debug_enabled
from whisper_bot.events import EventBus, PipelineError, TTSSpeaking, TTSSentenceStart, TTSSentenceDone
from whisper_bot.tts import TTSProvider


# ---------------------------------------------------------------------------
# Pipelined subprocess script — runs for the lifetime of each worker
# ---------------------------------------------------------------------------

def _build_pipelined_script() -> str:
    """Build the inline script for one Kokoro worker subprocess.

    The script speaks a two-phase protocol:

    **Input (stdin, JSONL):**
    - ``{"type":"speak","seq":0,"text":"...","voice":"af_heart"}``
      → generate audio for this sentence, store it, reply ``ready``
    - ``{"type":"play","seq":0}``
      → play the stored audio, reply ``done``
    - ``{"type":"shutdown"}``
      → exit

    **Output (stdout, JSONL):**
    - ``{"type":"ready","seq":0}``  — generation complete
    - ``{"type":"done","seq":0}``   — playback complete
    - ``{"type":"error","seq":0,"message":"..."}``
    """
    return r"""import os, sys, json, tempfile, subprocess as _sp
from pathlib import Path

_DEBUG = os.environ.get("WHISPER_BOT_DEBUG")
def _debug(msg):
    if _DEBUG:
        print(msg, file=sys.stderr, flush=True)

# ---- Eager init ----
_debug("Loading kokoro...")
from kokoro import KPipeline
import numpy as np

try:
    import sounddevice as _sd
    _HAS_SD = True
    _debug("Using sounddevice")
except ImportError:
    _HAS_SD = False
    import soundfile as _sf
    _debug("Using soundfile + afplay")

def _play(audio, sr=24000):
    if _HAS_SD:
        _stream = _sd.OutputStream(samplerate=sr, channels=1, blocksize=0)
        _stream.start()
        _stream.write(audio)
        _stream.stop()
        _stream.close()
    else:
        _tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            _sf.write(_tmp.name, audio, sr)
            _sp.run(["afplay", _tmp.name], check=True)
        finally:
            Path(_tmp.name).unlink(missing_ok=True)

# Eager init — load the model at startup
_voice = "af_heart"
_lang = "a"
_pipeline = KPipeline(lang_code=_lang)
_debug("Pipeline eagerly initialized")

_debug("Kokoro subprocess ready")

# Per-worker state
_pending_audio = None  # numpy array, set by "speak", consumed by "play"
_pending_seq = -1

for _line in sys.stdin:
    _line = _line.strip()
    if not _line:
        continue
    try:
        _msg = json.loads(_line)
    except json.JSONDecodeError:
        continue

    _type = _msg.get("type")

    if _type == "shutdown":
        _debug("Shutdown received")
        break

    elif _type == "speak":
        _text = _msg.get("text", "")
        _seq = _msg.get("seq", -1)
        _pending_seq = _seq

        if not _text.strip():
            _pending_audio = np.array([], dtype=np.float32)
            print(json.dumps({"type": "ready", "seq": _seq}), flush=True)
            continue

        _voice_override = _msg.get("voice", _voice)

        _debug(f"Generating audio for seq={_seq}: {len(_text)} chars")
        _audio_parts = []
        for _result in _pipeline(_text, voice=_voice_override):
            _ao = _result.audio
            if _ao is None:
                continue
            _audio_parts.append(
                _ao.cpu().numpy() if hasattr(_ao, "cpu") else np.asarray(_ao)
            )
        if _audio_parts:
            _pending_audio = np.concatenate(_audio_parts) if len(_audio_parts) > 1 else _audio_parts[0]
        else:
            _pending_audio = np.array([], dtype=np.float32)

        _debug(f"Generated {len(_pending_audio)} samples for seq={_seq}")
        print(json.dumps({"type": "ready", "seq": _seq}), flush=True)

    elif _type == "play":
        _seq = _msg.get("seq", -1)
        if _pending_audio is None or _pending_seq != _seq:
            _debug(f"ERROR: play for seq={_seq} but pending is seq={_pending_seq}")
            print(json.dumps({"type": "error", "seq": _seq, "message": "no pending audio"}), flush=True)
            continue

        _audio = _pending_audio
        _pending_audio = None

        if len(_audio) == 0:
            _debug(f"No audio for seq={_seq}, skipping playback")
            print(json.dumps({"type": "done", "seq": _seq}), flush=True)
            continue

        _debug(f"Playing {len(_audio)} samples for seq={_seq}")
        _play(_audio)
        _debug(f"Playback complete for seq={_seq}")
        print(json.dumps({"type": "done", "seq": _seq}), flush=True)

_debug("Kokoro subprocess exiting")
"""


# ---------------------------------------------------------------------------
# Streaming provider
# ---------------------------------------------------------------------------


class StreamingKokoroTTS(TTSProvider):
    """Sentence-queued Kokoro TTS backed by a pool of persistent subprocesses.

    A pool of N subprocess workers each load KPipeline once at startup.
    Incoming sentences are dispatched round-robin to free workers.  A
    two-phase protocol (speak → ready → play → done) lets sentence N+1
    generate while sentence N plays, hiding generation latency behind
    playback time.

    Implements ``speak(text)`` for backward compat (Phase 1 API), and
    adds ``speak_sentence(text)`` and ``wait()`` for streaming use.
    """

    # Type alias for per-worker state dict
    _WorkerState = dict

    def __init__(self, config: dict, event_bus: EventBus) -> None:
        tts_cfg = config.get("tts", {})
        self._voice: str = tts_cfg.get("voice", "af_heart")
        self._enabled: bool = tts_cfg.get("enabled", True)
        self._pool_size: int = tts_cfg.get("pool_size", 2)
        self._event_bus = event_bus

        # Queue of pending sentences + sentinel
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._done_event = asyncio.Event()
        self._done_event.set()
        self._started = False

        # Per-worker state
        # Each entry: {
        #   "proc": Process | None,
        #   "stderr_task": Task | None,
        #   "free": bool,        # True = ready for new work
        #   "pending_seq": int | None,  # which seq this worker is handling
        #   "ready_event": Event, # set when "Kokoro subprocess ready" seen
        # }
        self._workers: list[dict[str, Any]] = []

        # Dispatcher / sequencing state
        self._next_seq: int = 0          # next seq number to assign
        self._next_play_seq: int = 0     # next seq that should play
        self._pending_text: dict[int, str] = {}  # seq -> text (not dispatched yet)
        self._ready_seqs: set[int] = set()       # seqs where worker said "ready"
        self._done_seqs: set[int] = set()        # seqs where worker said "done"

        # Background tasks
        self._dispatcher_task: asyncio.Task[Any] | None = None

        _debug(
            f"[DEBUG streaming_tts] StreamingKokoroTTS initialized:"
            f" voice={self._voice} enabled={self._enabled}"
            f" pool_size={self._pool_size}"
        )

    # ------------------------------------------------------------------
    # Warmup — eager start
    # ------------------------------------------------------------------

    async def warmup(self, progress_callback: collections.abc.Callable[[str, str, float], Awaitable[None]] | None = None) -> None:
        """Eagerly start the Kokoro subprocess pool.

        Called at pipeline startup.  Blocks until all workers signal
        "Kokoro subprocess ready" on stderr, or a 120s timeout elapses.

        Args:
            progress_callback: Optional ``(provider, status, pct)`` for UI.
        """
        if not self._enabled:
            return
        _debug(f"[DEBUG streaming_tts] Warmup: starting {self._pool_size} workers...")

        self._workers = [None] * self._pool_size  # placeholder

        # Spawn workers, reporting progress for each
        for i in range(self._pool_size):
            if progress_callback is not None:
                await progress_callback(
                    "tts",
                    f"spawning worker {i+1}/{self._pool_size}...",
                    0.1 + 0.20 * i / self._pool_size,
                )
            await self._spawn_worker(i)

        # Wait for all workers to signal readiness, polling to update progress
        ready_count = 0
        deadline = asyncio.get_event_loop().time() + 120.0
        while ready_count < self._pool_size and asyncio.get_event_loop().time() < deadline:
            for i in range(self._pool_size):
                w = self._workers[i]
                if not w.get("_reported_ready", False) and w["ready_event"].is_set():
                    w["_reported_ready"] = True
                    ready_count += 1
                    _debug(f"[DEBUG streaming_tts] Worker {i} ready ({ready_count}/{self._pool_size})")
                    if progress_callback is not None:
                        await progress_callback(
                            "tts",
                            f"worker {i+1}/{self._pool_size} ready",
                            0.3 + 0.70 * ready_count / self._pool_size,
                        )
            await asyncio.sleep(0.2)

        for i in range(self._pool_size):
            if not self._workers[i]["ready_event"].is_set():
                _debug(f"[DEBUG streaming_tts] Worker {i} did not signal ready within 120s")

        _debug("[DEBUG streaming_tts] Warmup complete")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def speak(self, text: str) -> None:
        """Non-streaming mode: speak full text (backward compat).

        Splits *text* into sentences and queues them all at once, then
        waits for playback to complete.
        """
        if not self._enabled:
            return

        import re
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        for sentence in sentences:
            await self.speak_sentence(sentence)

        await self.wait()

    async def speak_sentence(self, text: str) -> None:
        """Queue a single sentence for TTS playback.

        Returns immediately.  The sentence is played in order when prior
        sentences have finished.
        """
        if not self._enabled or not text.strip():
            return

        # Start the dispatcher if not running
        if not self._started:
            self._started = True
            self._done_event.clear()
            self._dispatcher_task = asyncio.create_task(self._dispatcher())

        await self._queue.put(text)

    async def wait(self) -> None:
        """Wait for all queued sentences to finish playing.

        Sends a sentinel to the dispatcher.  The dispatcher will process
        all pending sentences first (FIFO) then exit.
        """
        if not self._started:
            return
        await self._queue.put(None)
        await self._done_event.wait()

    # ------------------------------------------------------------------
    # Dispatcher — sentence queue consumer + pool orchestrator
    # ------------------------------------------------------------------

    async def _dispatcher(self) -> None:
        """Background task: pull sentences from queue, assign seq numbers,
        dispatch to free workers, and coordinate play ordering."""
        _debug("[DEBUG streaming_tts] Dispatcher started")
        first_sentence = True
        try:
            while True:
                item = await self._queue.get()

                if item is None:
                    _debug("[DEBUG streaming_tts] Dispatcher received sentinel")
                    break

                if first_sentence:
                    await self._event_bus.emit(TTSSpeaking())
                    first_sentence = False

                # Assign sequence number
                seq = self._next_seq
                self._next_seq += 1
                self._pending_text[seq] = item

                # Emit event before dispatching to worker
                await self._event_bus.emit(TTSSentenceStart(text=item))

                # Try to dispatch to a free worker
                await self._dispatch_pending()

            # Sentinel received — wait for all dispatched work to complete
            while len(self._done_seqs) < self._next_seq:
                await asyncio.sleep(0.1)

        finally:
            self._done_event.set()
            self._started = False
            _debug("[DEBUG streaming_tts] Dispatcher exited (restartable)")

    async def _dispatch_pending(self) -> None:
        """Find the next undispatched sentence and send it to a free worker."""
        # Seqs already sent to worker are in ready_seqs or done_seqs
        dispatched = self._ready_seqs | self._done_seqs
        for seq in sorted(self._pending_text.keys()):
            if seq in dispatched:
                continue  # already sent to a worker

            # Find a free worker
            for i, w in enumerate(self._workers):
                if w and w["free"]:
                    text = self._pending_text[seq]
                    w["free"] = False
                    w["pending_seq"] = seq
                    await self._send_speak(i, seq, text)
                    return  # sent one, next call will send another

            # No free worker — will try again later
            return

    async def _maybe_send_play(self) -> None:
        """Send 'play' commands for all ready sentences whose turn has come.

        Only sends play for seq N after receiving done for seq N-1,
        guaranteeing sequential audio output regardless of which worker
        finishes generation first.
        """
        while self._next_play_seq in self._ready_seqs:
            seq = self._next_play_seq

            # Find which worker holds this seq
            sent = False
            for i, w in enumerate(self._workers):
                if w and w.get("pending_seq") == seq:
                    await self._send_play(i, seq)
                    sent = True
                    break

            if sent:
                self._ready_seqs.discard(seq)
                self._next_play_seq += 1
            else:
                _debug(f"[DEBUG streaming_tts] WARNING: no worker found for seq={seq}")
                self._done_seqs.add(seq)
                self._next_play_seq += 1

    # ------------------------------------------------------------------
    # Per-worker stdout reader
    # ------------------------------------------------------------------

    async def _reader(self, worker_idx: int) -> None:
        """Read stdout from a single worker, dispatching ready/done/error
        messages to the sequencing state machine."""
        w = self._workers[worker_idx]
        proc = w["proc"]
        _debug(f"[DEBUG streaming_tts] Reader {worker_idx} started")

        try:
            while True:
                line = await asyncio.wait_for(
                    proc.stdout.readline(),
                    timeout=300.0,
                )
                if not line:
                    _debug(f"[DEBUG streaming_tts] EOF on worker {worker_idx} stdout — process died")
                    break

                decoded = line.decode(errors="replace").strip()
                _debug(f"[DEBUG streaming_tts] Worker {worker_idx} stdout: {decoded[:200]}")

                try:
                    msg = json.loads(decoded)
                except json.JSONDecodeError:
                    _debug(f"[DEBUG streaming_tts] Worker {worker_idx} bad JSON: {decoded[:100]}")
                    continue

                msg_type = msg.get("type")
                seq = msg.get("seq", -1)

                if msg_type == "ready":
                    self._ready_seqs.add(seq)
                    # Check if this seq can play now
                    await self._maybe_send_play()

                elif msg_type == "done":
                    self._done_seqs.add(seq)
                    # Mark worker as free and dispatch more work
                    self._workers[worker_idx]["free"] = True
                    self._workers[worker_idx]["pending_seq"] = None
                    await self._dispatch_pending()
                    # Emit done event
                    await self._event_bus.emit(TTSSentenceDone())
                    # Check if next seq can play
                    await self._maybe_send_play()

                elif msg_type == "error":
                    _debug(f"[DEBUG streaming_tts] Worker {worker_idx} error for seq={seq}: {msg.get('message', '?')}")
                    self._done_seqs.add(seq)
                    self._workers[worker_idx]["free"] = True
                    self._workers[worker_idx]["pending_seq"] = None
                    await self._dispatch_pending()
                    await self._event_bus.emit(TTSSentenceDone())
                    await self._maybe_send_play()

        except asyncio.TimeoutError:
            _debug(f"[DEBUG streaming_tts] Reader {worker_idx} timeout — worker may be hung")
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            _debug(f"[DEBUG streaming_tts] Reader {worker_idx} error: {type(exc).__name__}: {exc}")

        # Worker died — mark free if seq was pending
        failed_seq = self._workers[worker_idx].get("pending_seq")
        self._workers[worker_idx]["free"] = True
        self._workers[worker_idx]["pending_seq"] = None

        if failed_seq is not None and failed_seq not in self._done_seqs:
            _debug(f"[DEBUG streaming_tts] Worker {worker_idx} died with pending seq={failed_seq}")
            # Re-spawn and re-dispatch the failed sentence
            await self._respawn_worker(worker_idx)
            text = self._pending_text.get(failed_seq)
            if text and failed_seq not in self._done_seqs:
                self._workers[worker_idx]["free"] = False
                self._workers[worker_idx]["pending_seq"] = failed_seq
                await self._send_speak(worker_idx, failed_seq, text)
        else:
            # Just re-spawn the dead worker
            await self._respawn_worker(worker_idx)

    # ------------------------------------------------------------------
    # Worker lifecycle
    # ------------------------------------------------------------------

    async def _spawn_worker(self, idx: int) -> None:
        """Spawn one persistent Kokoro subprocess worker."""
        script = _build_pipelined_script()
        _debug(f"[DEBUG streaming_tts] Spawning worker {idx} (script={len(script)} chars)")

        env = None
        if _debug_enabled:
            env = {**os.environ, "WHISPER_BOT_DEBUG": "1"}

        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-c",
                script,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
            _debug(f"[DEBUG streaming_tts] Worker {idx} spawned, pid={proc.pid}")

            ready_ev = asyncio.Event()
            stderr_task = asyncio.create_task(self._drain_stderr(idx, proc, ready_ev))

            self._workers[idx] = {
                "proc": proc,
                "stderr_task": stderr_task,
                "free": True,
                "pending_seq": None,
                "ready_event": ready_ev,
            }

            # Start the stdout reader for this worker
            asyncio.create_task(self._reader(idx))

        except Exception as exc:
            _debug(f"[DEBUG streaming_tts] Failed to spawn worker {idx}: {exc}")
            self._workers[idx] = {
                "proc": None,
                "stderr_task": None,
                "free": True,
                "pending_seq": None,
                "ready_event": asyncio.Event(),
            }

    async def _respawn_worker(self, idx: int) -> None:
        """Re-spawn a dead worker and restore it to the pool."""
        _debug(f"[DEBUG streaming_tts] Respawning worker {idx}...")

        # Clean up old process resources
        old = self._workers[idx]
        if old.get("stderr_task"):
            old["stderr_task"].cancel()
            try:
                await old["stderr_task"]
            except asyncio.CancelledError:
                pass

        # Kill the old process if somehow still alive
        if old.get("proc") and old["proc"].returncode is None:
            try:
                old["proc"].kill()
                await old["proc"].wait()
            except Exception:
                pass

        await self._spawn_worker(idx)

    async def _drain_stderr(self, idx: int, proc: asyncio.subprocess.Process,
                            ready_event: asyncio.Event) -> None:
        """Read stderr from *proc* and forward to debug log.

        Sets *ready_event* when "Kokoro subprocess ready" is seen.
        """
        assert proc.stderr is not None
        try:
            while True:
                line = await proc.stderr.readline()
                if not line:
                    break
                text = line.decode(errors="replace").strip()
                if text:
                    _debug(f"[DEBUG tts proc {idx}] {text}")
                    if "Kokoro subprocess ready" in text:
                        ready_event.set()
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Inter-worker communication helpers
    # ------------------------------------------------------------------

    async def _send_speak(self, worker_idx: int, seq: int, text: str) -> None:
        """Send a ``speak`` command to *worker_idx*."""
        w = self._workers[worker_idx]
        proc = w["proc"]
        if proc is None or proc.returncode is not None:
            _debug(f"[DEBUG streaming_tts] Worker {worker_idx} dead, cannot send speak for seq={seq}")
            return

        payload = json.dumps({
            "type": "speak",
            "seq": seq,
            "text": text,
            "voice": self._voice,
        }) + "\n"

        _debug(f"[DEBUG streaming_tts] -> Worker {worker_idx}: speak seq={seq} ({len(text)} chars)")
        proc.stdin.write(payload.encode("utf-8"))
        await proc.stdin.drain()

    async def _send_play(self, worker_idx: int, seq: int) -> None:
        """Send a ``play`` command to *worker_idx*."""
        w = self._workers[worker_idx]
        proc = w["proc"]
        if proc is None or proc.returncode is not None:
            _debug(f"[DEBUG streaming_tts] Worker {worker_idx} dead, cannot send play for seq={seq}")
            return

        payload = json.dumps({"type": "play", "seq": seq}) + "\n"
        _debug(f"[DEBUG streaming_tts] -> Worker {worker_idx}: play seq={seq}")
        proc.stdin.write(payload.encode("utf-8"))
        await proc.stdin.drain()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Cancel dispatcher, kill all workers, and clean up."""
        # 1. Cancel the dispatcher
        if self._dispatcher_task:
            self._dispatcher_task.cancel()
            try:
                await self._dispatcher_task
            except asyncio.CancelledError:
                pass
            self._dispatcher_task = None
        self._started = False
        self._done_event.set()

        # 2. Send shutdown sentinel to all workers
        for i, w in enumerate(self._workers):
            proc = w.get("proc")
            if proc is not None and proc.returncode is None:
                try:
                    _debug(f"[DEBUG streaming_tts] Sending shutdown to worker {i}, pid={proc.pid}")
                    assert proc.stdin is not None
                    proc.stdin.write(b'{"type":"shutdown"}\n')
                    await proc.stdin.drain()
                except Exception:
                    pass

        # 3. Wait briefly for graceful exit, then kill stragglers
        for i, w in enumerate(self._workers):
            proc = w.get("proc")
            if proc is None:
                continue
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
                _debug(f"[DEBUG streaming_tts] Worker {i} exited cleanly")
            except asyncio.TimeoutError:
                _debug(f"[DEBUG streaming_tts] Worker {i} did not exit — sending SIGKILL")
                try:
                    proc.send_signal(signal.SIGKILL)
                    await proc.wait()
                except ProcessLookupError:
                    pass
                except Exception:
                    pass
            except ProcessLookupError:
                pass
            except Exception as exc:
                _debug(f"[DEBUG streaming_tts] Error shutting down worker {i}: {exc}")

            # Cancel stderr drainer
            stderr_task = w.get("stderr_task")
            if stderr_task:
                stderr_task.cancel()
                try:
                    await stderr_task
                except asyncio.CancelledError:
                    pass

        self._workers.clear()
        self._ready_seqs.clear()
        self._done_seqs.clear()
        self._pending_text.clear()
        self._next_seq = 0
        self._next_play_seq = 0
        _debug("[DEBUG streaming_tts] Shutdown complete")
