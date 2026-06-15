"""Piper TTS provider — local neural TTS via the ``piper-tts`` Python package.

Uses `Piper <https://github.com/rhasspy/piper>`_ for fast CPU-based TTS.
Supports both one-shot (non-streaming) and sentence-streaming modes.

The streaming provider loads ``PiperVoice`` directly in worker subprocesses
for efficient synthesis (no CLI subprocess per sentence).

Voice model ``.onnx`` files go in ``~/.whisper-bot/piper/`` by default.
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
from whisper_bot.events import (
    EventBus,
    PipelineError,
    TTSSpeaking,
    TTSDone,
    TTSSentenceStart,
    TTSSentenceDone,
)
from whisper_bot.tts import TTSProvider


# ---------------------------------------------------------------------------
# Pipelined subprocess script — runs for the lifetime of each worker
# ---------------------------------------------------------------------------


def _build_pipelined_script(model_path: str, speed: float = 1.0) -> str:
    """Build the inline script for one Piper worker subprocess.

    The script uses ``piper.voice.PiperVoice`` directly (no CLI subprocess)
    and speaks a two-phase protocol identical to ``StreamingKokoroTTS`` workers:

    **Input (stdin, JSONL):**
    - ``{"type":"speak","seq":0,"text":"...","speed":1.0}``
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
    # Safely embed the model path into the inline script
    safe_path = model_path.replace("\\", "\\\\").replace("'", "\\'")
    return f'''import sys, json, os, tempfile, subprocess as _sp
from pathlib import Path
import numpy as np

_DEBUG = os.environ.get("WHISPER_BOT_DEBUG")
def _debug(msg):
    if _DEBUG:
        print(msg, file=sys.stderr, flush=True)

_MODEL_PATH = '{safe_path}'
_DEFAULT_SPEED = {speed}

try:
    import sounddevice as _sd
    _HAS_SD = True
except ImportError:
    _HAS_SD = False
    import soundfile as _sf
    _debug("sounddevice not available, using soundfile+afplay")

def _resample(audio, spd):
    if spd == 1.0 or len(audio) == 0:
        return audio
    _n_orig = len(audio)
    _n_new = max(1, int(_n_orig / spd))
    _idx = np.linspace(0, _n_orig - 1, _n_new)
    _fl = np.floor(_idx).astype(int)
    _cl = np.minimum(_fl + 1, _n_orig - 1)
    _fr = _idx - _fl
    return audio[_fl] * (1.0 - _fr) + audio[_cl] * _fr

def _play(audio, sr=22050, spd=1.0):
    audio = _resample(audio, spd)
    if _HAS_SD:
        _stream = _sd.OutputStream(samplerate=sr, channels=1, blocksize=0)
        _stream.start()
        _stream.write(audio.astype(np.float32))
        _stream.stop()
        _stream.close()
    else:
        _tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        try:
            _sf.write(_tmp.name, audio, sr)
            _sp.run(["afplay", _tmp.name], check=True)
        finally:
            Path(_tmp.name).unlink(missing_ok=True)

# ---- Eager model load ----
_debug("Loading Piper model...")
try:
    from piper.voice import PiperVoice
    _VOICE = PiperVoice.load(_MODEL_PATH)
    _SR = _VOICE.config.sample_rate
    _debug(f"Piper model loaded (sr={{_SR}})")
except Exception as _e:
    _debug(f"FATAL: cannot load Piper model: {{_e}}")
    print(json.dumps({{"type": "error", "seq": -1, "message": f"Model load failed: {{_e}}"}}), flush=True)
    sys.exit(1)

_debug("Piper subprocess ready")

_pending_audio = None
_pending_seq = -1
_pending_speed = _DEFAULT_SPEED

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
        _pending_speed = _msg.get("speed", _DEFAULT_SPEED)

        if not _text.strip():
            _pending_audio = np.array([], dtype=np.float32)
            print(json.dumps({{"type": "ready", "seq": _seq}}), flush=True)
            continue

        _debug(f"Generating audio for seq={{_seq}}: {{len(_text)}} chars")

        try:
            _chunks = []
            for _chunk in _VOICE.synthesize(_text):
                _chunks.append(_chunk.audio_float_array)
            if _chunks:
                _pending_audio = np.concatenate(_chunks) if len(_chunks) > 1 else _chunks[0]
            else:
                _pending_audio = np.array([], dtype=np.float32)
        except Exception as _e:
            _debug(f"Piper synthesis error: {{_e}}")
            _pending_audio = np.array([], dtype=np.float32)
            print(json.dumps({{"type": "error", "seq": _seq, "message": str(_e)}}), flush=True)
            continue

        _debug(f"Generated {{len(_pending_audio)}} samples for seq={{_seq}}")
        print(json.dumps({{"type": "ready", "seq": _seq}}), flush=True)

    elif _type == "play":
        _seq = _msg.get("seq", -1)
        if _pending_audio is None or _pending_seq != _seq:
            _debug(f"ERROR: play for seq={{_seq}} but pending is seq={{_pending_seq}}")
            print(json.dumps({{"type": "error", "seq": _seq, "message": "no pending audio"}}), flush=True)
            continue

        _audio = _pending_audio
        _pending_audio = None
        _play_speed = _pending_speed

        if len(_audio) == 0:
            print(json.dumps({{"type": "done", "seq": _seq}}), flush=True)
            continue

        _debug(f"Playing {{len(_audio)}} samples for seq={{_seq}}")
        _play(_audio, _SR, _play_speed)
        print(json.dumps({{"type": "done", "seq": _seq}}), flush=True)

_debug("Piper subprocess exiting")
'''


# ---------------------------------------------------------------------------
# Non-streaming provider
# ---------------------------------------------------------------------------


class PiperTTS(TTSProvider):
    """Non-streaming Piper TTS — spawns a piper subprocess per call.

    Reads the model path from ``config["tts"]["model_path"]`` (default:
    ``~/.whisper-bot/piper/<voice>.onnx``).  Falls back gracefully via
    ``PipelineError`` events if the CLI or model file is missing.
    """

    def __init__(self, config: dict, event_bus: EventBus) -> None:
        tts_cfg = config.get("tts", {})
        self._enabled: bool = tts_cfg.get("enabled", True)
        self._event_bus = event_bus
        self._model_path = _resolve_model_path(config)
        self._speed: float = tts_cfg.get("speed", 1.0)
        _debug(
            f"[DEBUG piper] PiperTTS initialized:"
            f" enabled={self._enabled}"
            f" speed={self._speed}"
            f" model={self._model_path}"
        )

    async def speak(self, text: str) -> None:
        """Speak *text* via Piper (subprocess per call)."""
        if not self._enabled:
            return
        await self._event_bus.emit(TTSSpeaking())
        ok = await self._run_piper(text)
        if ok:
            await self._event_bus.emit(TTSDone())

    async def _run_piper(self, text: str) -> bool:
        """Run the ``piper`` CLI to generate and play audio."""
        env = None
        if _debug_enabled:
            env = {**os.environ, "WHISPER_BOT_DEBUG": "1"}

        try:
            proc = await asyncio.create_subprocess_exec(
                "piper",
                "--model", self._model_path,
                "--output-raw",
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
            stdout, stderr = await proc.communicate(text.encode("utf-8"))

            if proc.returncode != 0:
                err_msg = (
                    stderr.decode().strip()
                    or f"Piper exited with code {proc.returncode}"
                )
                await self._event_bus.emit(
                    PipelineError(stage="tts", fatal=False, message=err_msg)
                )
                return False

            if not stdout:
                return True  # empty text

            import sounddevice as sd
            import numpy as np

            audio = np.frombuffer(stdout, dtype=np.int16).astype(np.float32) / 32768.0
            # Apply speed via resampling
            if self._speed != 1.0 and len(audio) > 0:
                n_orig = len(audio)
                n_new = max(1, int(n_orig / self._speed))
                indices = np.linspace(0, n_orig - 1, n_new)
                floor_idx = np.floor(indices).astype(int)
                ceil_idx = np.minimum(floor_idx + 1, n_orig - 1)
                frac = indices - floor_idx
                audio = audio[floor_idx] * (1.0 - frac) + audio[ceil_idx] * frac
            sd.play(audio, 22050)
            sd.wait()
            return True

        except FileNotFoundError:
            _debug("[DEBUG piper] piper CLI not found")
            await self._event_bus.emit(
                PipelineError(
                    stage="tts",
                    fatal=False,
                    message="piper CLI not found.  Install it: pip install piper-tts",
                )
            )
            return False
        except Exception as exc:
            _debug(f"[DEBUG piper] Error: {type(exc).__name__}: {exc}")
            await self._event_bus.emit(
                PipelineError(stage="tts", fatal=False, message=str(exc))
            )
            return False


# ---------------------------------------------------------------------------
# Streaming provider (sentence-level pipelining via worker pool)
# ---------------------------------------------------------------------------


class StreamingPiperTTS(TTSProvider):
    """Sentence-streaming Piper TTS backed by a pool of persistent subprocesses.

    Architecture mirrors :class:`StreamingKokoroTTS`:

    * A pool of N subprocess workers each load ``PiperVoice`` once at startup.
    * Incoming sentences are dispatched round-robin to free workers.
    * A two-phase protocol (``speak`` → ``ready`` → ``play`` → ``done``)
      lets sentence N+1 generate while sentence N plays.

    Implements ``speak(text)`` for backward compat and adds
    ``speak_sentence(text)`` / ``wait()`` for streaming use.
    """

    _WorkerState = dict

    def __init__(self, config: dict, event_bus: EventBus) -> None:
        self._enabled: bool = config.get("tts", {}).get("enabled", True)
        self._pool_size: int = config.get("tts", {}).get("pool_size", 2)
        self._model_path = _resolve_model_path(config)
        self._speed: float = config.get("tts", {}).get("speed", 1.0)
        self._event_bus = event_bus

        # Queue of pending sentences + sentinel
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._done_event = asyncio.Event()
        self._done_event.set()
        self._started = False

        # Per-worker state
        self._workers: list[dict[str, Any]] = []

        # Dispatcher / sequencing state
        self._next_seq: int = 0
        self._next_play_seq: int = 0
        self._pending_text: dict[int, str] = {}
        self._ready_seqs: set[int] = set()
        self._done_seqs: set[int] = set()

        # Background tasks
        self._dispatcher_task: asyncio.Task[Any] | None = None

        _debug(
            f"[DEBUG streaming_piper] StreamingPiperTTS initialized:"
            f" enabled={self._enabled}"
            f" pool_size={self._pool_size}"
            f" model={self._model_path}"
        )

    # ------------------------------------------------------------------
    # Warmup — eager start
    # ------------------------------------------------------------------

    async def warmup(
        self,
        progress_callback: collections.abc.Callable[
            [str, str, float], Awaitable[None]
        ]
        | None = None,
    ) -> None:
        """Eagerly start the Piper subprocess pool.

        Blocks until all workers signal *Piper subprocess ready* on stderr,
        or a 120 s timeout elapses.
        """
        if not self._enabled:
            return
        _debug(
            f"[DEBUG streaming_piper] Warmup: starting {self._pool_size} workers..."
        )

        self._workers = [None] * self._pool_size

        for i in range(self._pool_size):
            if progress_callback is not None:
                await progress_callback(
                    "tts",
                    f"spawning worker {i+1}/{self._pool_size}...",
                    0.1 + 0.20 * i / self._pool_size,
                )
            await self._spawn_worker(i)

        # Wait for all workers to signal readiness
        ready_count = 0
        deadline = asyncio.get_event_loop().time() + 120.0
        while (
            ready_count < self._pool_size
            and asyncio.get_event_loop().time() < deadline
        ):
            for i in range(self._pool_size):
                w = self._workers[i]
                if not w.get("_reported_ready", False) and w["ready_event"].is_set():
                    w["_reported_ready"] = True
                    ready_count += 1
                    _debug(
                        f"[DEBUG streaming_piper] Worker {i} ready"
                        f" ({ready_count}/{self._pool_size})"
                    )
                    if progress_callback is not None:
                        await progress_callback(
                            "tts",
                            f"worker {i+1}/{self._pool_size} ready",
                            0.3 + 0.70 * ready_count / self._pool_size,
                        )
            await asyncio.sleep(0.2)

        for i in range(self._pool_size):
            if not self._workers[i]["ready_event"].is_set():
                _debug(
                    f"[DEBUG streaming_piper] Worker {i} did not signal ready"
                    " within 120s"
                )

        _debug("[DEBUG streaming_piper] Warmup complete")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def speak(self, text: str) -> None:
        """Non-streaming mode: split into sentences, queue all, wait."""
        if not self._enabled:
            return
        import re

        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", text)
            if s.strip()
        ]
        for sentence in sentences:
            await self.speak_sentence(sentence)
        await self.wait()

    async def speak_sentence(self, text: str) -> None:
        """Queue a single sentence for TTS playback (returns immediately)."""
        if not self._enabled or not text.strip():
            return
        if not self._started:
            self._started = True
            self._done_event.clear()
            self._dispatcher_task = asyncio.create_task(self._dispatcher())
        await self._queue.put(text)

    async def wait(self) -> None:
        """Wait for all queued sentences to finish playing."""
        if not self._started:
            return
        await self._queue.put(None)
        await self._done_event.wait()

    # ------------------------------------------------------------------
    # Dispatcher
    # ------------------------------------------------------------------

    async def _dispatcher(self) -> None:
        """Background task: pull sentences from queue and dispatch to pool."""
        _debug("[DEBUG streaming_piper] Dispatcher started")
        try:
            while True:
                item = await self._queue.get()
                if item is None:
                    _debug("[DEBUG streaming_piper] Dispatcher received sentinel")
                    break

                seq = self._next_seq
                self._next_seq += 1
                self._pending_text[seq] = item
                await self._event_bus.emit(TTSSentenceStart(text=item))
                await self._dispatch_pending()

            while len(self._done_seqs) < self._next_seq:
                await asyncio.sleep(0.1)
        finally:
            self._done_event.set()
            self._started = False
            _debug("[DEBUG streaming_piper] Dispatcher exited (restartable)")

    async def _dispatch_pending(self) -> None:
        """Dispatch the next undispatched sentence to a free worker."""
        dispatched = self._ready_seqs | self._done_seqs
        for seq in sorted(self._pending_text.keys()):
            if seq in dispatched:
                continue
            for i, w in enumerate(self._workers):
                if w and w["free"] and w["proc"] is not None:
                    text = self._pending_text[seq]
                    w["free"] = False
                    w["pending_seq"] = seq
                    await self._send_speak(i, seq, text)
                    return
            return

    async def _maybe_send_play(self) -> None:
        """Send play commands for ready sentences whose turn has come."""
        while self._next_play_seq in self._ready_seqs:
            seq = self._next_play_seq
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
                _debug(
                    f"[DEBUG streaming_piper] WARNING: no worker for seq={seq}"
                )
                self._done_seqs.add(seq)
                self._next_play_seq += 1

    # ------------------------------------------------------------------
    # Per-worker stdout reader
    # ------------------------------------------------------------------

    async def _reader(self, worker_idx: int) -> None:
        """Read stdout from a worker, dispatching ready/done/error messages."""
        w = self._workers[worker_idx]
        proc = w["proc"]
        _debug(f"[DEBUG streaming_piper] Reader {worker_idx} started")

        try:
            while True:
                line = await asyncio.wait_for(
                    proc.stdout.readline(), timeout=300.0
                )
                if not line:
                    _debug(
                        f"[DEBUG streaming_piper] EOF on worker {worker_idx}"
                        " stdout"
                    )
                    break

                decoded = line.decode(errors="replace").strip()
                _debug(
                    f"[DEBUG streaming_piper] Worker {worker_idx} stdout:"
                    f" {decoded[:200]}"
                )

                try:
                    msg = json.loads(decoded)
                except json.JSONDecodeError:
                    continue

                msg_type = msg.get("type")
                seq = msg.get("seq", -1)

                if msg_type == "ready":
                    self._ready_seqs.add(seq)
                    await self._maybe_send_play()
                elif msg_type == "done":
                    self._done_seqs.add(seq)
                    self._workers[worker_idx]["free"] = True
                    self._workers[worker_idx]["pending_seq"] = None
                    await self._dispatch_pending()
                    await self._event_bus.emit(TTSSentenceDone())
                    await self._maybe_send_play()
                elif msg_type == "error":
                    _debug(
                        f"[DEBUG streaming_piper] Worker {worker_idx} error"
                        f" seq={seq}: {msg.get('message', '?')}"
                    )
                    self._done_seqs.add(seq)
                    self._workers[worker_idx]["free"] = True
                    self._workers[worker_idx]["pending_seq"] = None
                    await self._dispatch_pending()
                    await self._event_bus.emit(TTSSentenceDone())
                    await self._maybe_send_play()

        except asyncio.TimeoutError:
            _debug(
                f"[DEBUG streaming_piper] Reader {worker_idx} timeout"
                " — worker may be hung"
            )
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            _debug(
                f"[DEBUG streaming_piper] Reader {worker_idx} error:"
                f" {type(exc).__name__}: {exc}"
            )

        # Worker died — mark free
        failed_seq = self._workers[worker_idx].get("pending_seq")
        self._workers[worker_idx]["free"] = True
        self._workers[worker_idx]["pending_seq"] = None

        # Don't respawn if the model is gone — would infinite-loop
        if not os.path.isfile(self._model_path):
            _debug(
                f"[DEBUG streaming_piper] Piper model missing,"
                f" skipping respawn for worker {worker_idx}"
            )
            await self._event_bus.emit(
                PipelineError(
                    stage="tts",
                    fatal=True,
                    message=f"Piper model not found: {self._model_path}",
                )
            )
            return

        if failed_seq is not None and failed_seq not in self._done_seqs:
            _debug(
                f"[DEBUG streaming_piper] Worker {worker_idx} died with"
                f" pending seq={failed_seq}"
            )
            await self._respawn_worker(worker_idx)
            text = self._pending_text.get(failed_seq)
            if text and failed_seq not in self._done_seqs:
                self._workers[worker_idx]["free"] = False
                self._workers[worker_idx]["pending_seq"] = failed_seq
                await self._send_speak(worker_idx, failed_seq, text)
        else:
            await self._respawn_worker(worker_idx)

    # ------------------------------------------------------------------
    # Worker lifecycle
    # ------------------------------------------------------------------

    async def _spawn_worker(self, idx: int) -> None:
        """Spawn one persistent Piper subprocess worker."""
        # Verify model exists before spawning
        if not os.path.isfile(self._model_path):
            _debug(
                f"[DEBUG streaming_piper] Piper model not found:"
                f" {self._model_path}"
            )
            await self._event_bus.emit(
                PipelineError(
                    stage="tts",
                    fatal=False,
                    message=f"Piper model not found: {self._model_path}",
                )
            )
            self._workers[idx] = {
                "proc": None,
                "stderr_task": None,
                "free": True,
                "pending_seq": None,
                "ready_event": asyncio.Event(),
                "_reported_ready": False,
            }
            return

        script = _build_pipelined_script(self._model_path, speed=self._speed)
        _debug(
            f"[DEBUG streaming_piper] Spawning worker {idx}"
            f" (script={len(script)} chars)"
        )

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
            _debug(
                f"[DEBUG streaming_piper] Worker {idx} spawned, pid={proc.pid}"
            )

            ready_ev = asyncio.Event()
            stderr_task = asyncio.create_task(
                self._drain_stderr(idx, proc, ready_ev)
            )

            self._workers[idx] = {
                "proc": proc,
                "stderr_task": stderr_task,
                "free": True,
                "pending_seq": None,
                "ready_event": ready_ev,
                "_reported_ready": False,
            }

            asyncio.create_task(self._reader(idx))

        except Exception as exc:
            _debug(
                f"[DEBUG streaming_piper] Failed to spawn worker {idx}: {exc}"
            )
            self._workers[idx] = {
                "proc": None,
                "stderr_task": None,
                "free": True,
                "pending_seq": None,
                "ready_event": asyncio.Event(),
                "_reported_ready": False,
            }

    async def _respawn_worker(self, idx: int) -> None:
        """Re-spawn a dead worker."""
        _debug(f"[DEBUG streaming_piper] Respawning worker {idx}...")
        old = self._workers[idx]
        if old.get("stderr_task"):
            old["stderr_task"].cancel()
            try:
                await old["stderr_task"]
            except asyncio.CancelledError:
                pass
        if old.get("proc") and old["proc"].returncode is None:
            try:
                old["proc"].kill()
                await old["proc"].wait()
            except Exception:
                pass
        await self._spawn_worker(idx)

    async def _drain_stderr(
        self,
        idx: int,
        proc: asyncio.subprocess.Process,
        ready_event: asyncio.Event,
    ) -> None:
        """Read stderr from the worker and set *ready_event* when ready."""
        assert proc.stderr is not None
        try:
            while True:
                line = await proc.stderr.readline()
                if not line:
                    break
                text = line.decode(errors="replace").strip()
                if text:
                    _debug(f"[DEBUG piper proc {idx}] {text}")
                    if "Piper subprocess ready" in text:
                        ready_event.set()
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Inter-worker communication helpers
    # ------------------------------------------------------------------

    async def _send_speak(
        self, worker_idx: int, seq: int, text: str
    ) -> None:
        """Send a ``speak`` command to *worker_idx*."""
        w = self._workers[worker_idx]
        proc = w["proc"]
        if proc is None or proc.returncode is not None:
            _debug(
                f"[DEBUG streaming_piper] Worker {worker_idx} dead,"
                f" cannot send speak for seq={seq} — recovering"
            )
            w["free"] = True
            w["pending_seq"] = None
            self._done_seqs.add(seq)
            await self._dispatch_pending()
            await self._maybe_send_play()
            return

        payload = (
            json.dumps({"type": "speak", "seq": seq, "text": text, "speed": self._speed}) + "\n"
        )
        _debug(
            f"[DEBUG streaming_piper] -> Worker {worker_idx}: speak seq={seq}"
            f" ({len(text)} chars, speed={self._speed})"
        )
        proc.stdin.write(payload.encode("utf-8"))
        await proc.stdin.drain()

    async def _send_play(self, worker_idx: int, seq: int) -> None:
        """Send a ``play`` command to *worker_idx*."""
        w = self._workers[worker_idx]
        proc = w["proc"]
        if proc is None or proc.returncode is not None:
            _debug(
                f"[DEBUG streaming_piper] Worker {worker_idx} dead,"
                f" cannot send play for seq={seq} — recovering"
            )
            w["free"] = True
            w["pending_seq"] = None
            self._done_seqs.add(seq)
            await self._dispatch_pending()
            return

        payload = json.dumps({"type": "play", "seq": seq}) + "\n"
        _debug(
            f"[DEBUG streaming_piper] -> Worker {worker_idx}: play seq={seq}"
        )
        proc.stdin.write(payload.encode("utf-8"))
        await proc.stdin.drain()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def shutdown(self) -> None:
        """Cancel dispatcher, kill all workers, clean up."""
        if self._dispatcher_task:
            self._dispatcher_task.cancel()
            try:
                await self._dispatcher_task
            except asyncio.CancelledError:
                pass
            self._dispatcher_task = None
        self._started = False
        self._done_event.set()

        for i, w in enumerate(self._workers):
            proc = w.get("proc")
            if proc is not None and proc.returncode is None:
                try:
                    _debug(
                        f"[DEBUG streaming_piper] Sending shutdown to"
                        f" worker {i}, pid={proc.pid}"
                    )
                    assert proc.stdin is not None
                    proc.stdin.write(b'{"type":"shutdown"}\n')
                    await proc.stdin.drain()
                except Exception:
                    pass

        for i, w in enumerate(self._workers):
            proc = w.get("proc")
            if proc is None:
                continue
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
                _debug(
                    f"[DEBUG streaming_piper] Worker {i} exited cleanly"
                )
            except asyncio.TimeoutError:
                _debug(
                    f"[DEBUG streaming_piper] Worker {i} did not exit"
                    " — sending SIGKILL"
                )
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
                _debug(
                    f"[DEBUG streaming_piper] Error shutting down"
                    f" worker {i}: {exc}"
                )

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
        _debug("[DEBUG streaming_piper] Shutdown complete")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _resolve_model_path(config: dict) -> str:
    """Resolve the Piper model path from config.

    Priority:
    1. ``config["tts"]["model_path"]`` (explicit path)
    2. ``~/.whisper-bot/piper/<voice>.onnx``
    """
    tts_cfg = config.get("tts", {})
    explicit = tts_cfg.get("model_path")
    if explicit:
        return os.path.expanduser(explicit)
    voice = tts_cfg.get("voice", "en_US-lessac-medium")
    return os.path.expanduser(f"~/.whisper-bot/piper/{voice}.onnx")
