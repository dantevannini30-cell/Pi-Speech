"""Kokoro TTS provider.

Uses the Kokoro TTS engine via subprocess for GIL isolation
(PyTorch holds the GIL during inference, so spawning a subprocess keeps the
main event loop responsive).
"""

from __future__ import annotations
from whisper_bot.debug import log as _debug
from whisper_bot.debug import _enabled as _debug_enabled

import asyncio
import os
import subprocess
import sys

from whisper_bot.events import PipelineError, TTSSpeaking, TTSDone
from whisper_bot.events.bus import EventBus
from whisper_bot.tts import TTSProvider


class KokoroTTS(TTSProvider):
    """Text-to-speech backed by Kokoro_.

    .. _Kokoro: https://github.com/hexgrad/kokoro
    """

    def __init__(self, config: dict, event_bus: EventBus) -> None:
        tts_cfg = config.get("tts", {})
        self._voice: str = tts_cfg.get("voice", "af_heart")
        self._enabled: bool = tts_cfg.get("enabled", True)
        self._speed: float = tts_cfg.get("speed", 1.0)
        self._event_bus = event_bus
        _debug(f"[DEBUG tts] KokoroTTS initialized: voice={self._voice} enabled={self._enabled} speed={self._speed}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def speak(self, text: str) -> None:
        """Speak *text* aloud via Kokoro.

        Spawns a short-lived Python subprocess that loads Kokoro, generates
        audio for the given text, and plays it.  If TTS is disabled or the
        subprocess fails the method returns gracefully after emitting the
        appropriate events.
        """
        _debug(f"[DEBUG tts] speak() called with ({len(text)} chars)")
        _debug(f"[DEBUG tts] Text: {text[:200]!r}")

        if not self._enabled:
            _debug("[DEBUG tts] TTS is disabled, returning immediately")
            return

        await self._event_bus.emit(TTSSpeaking())
        _debug("[DEBUG tts] TTSSpeaking event emitted")

        ok = await self._run_kokoro(text)

        if not ok:
            _debug("[DEBUG tts] _run_kokoro returned False, not emitting TTSDone")
            return  # PipelineError already emitted by _run_kokoro

        _debug("[DEBUG tts] _run_kokoro succeeded, emitting TTSDone")
        await self._event_bus.emit(TTSDone())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _run_kokoro(self, text: str) -> bool:
        """Run the Kokoro inline script as a subprocess.

        Returns ``True`` on success, ``False`` on failure (and emits
        ``PipelineError``).
        """
        return await _run_kokoro_process(text, self._voice, self._event_bus, speed=self._speed)


# ------------------------------------------------------------------
# Module-level helpers (shared by KokoroTTS and StreamingKokoroTTS)
# ------------------------------------------------------------------


async def _run_kokoro_process(text: str, voice: str, event_bus: EventBus, speed: float = 1.0) -> bool:
    """Run the Kokoro inline script as a subprocess.

    Returns ``True`` on success, ``False`` on failure (and emits
    ``PipelineError``).
    """
    kokoro_script = _build_kokoro_script(voice, speed=speed)

    _debug(f"[DEBUG tts] Spawning subprocess: {sys.executable} -c <inline script>")
    _debug(f"[DEBUG tts] Inline script length: {len(kokoro_script)} chars")

    env = None
    if _debug_enabled:
        env = {**os.environ, "WHISPER_BOT_DEBUG": "1"}

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        kokoro_script,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )
    _debug(f"[DEBUG tts] Subprocess spawned with pid={proc.pid}")

    _debug(f"[DEBUG tts] Sending {len(text)} bytes to subprocess stdin and waiting...")
    stdout, stderr = await proc.communicate(text.encode("utf-8"))
    _debug(f"[DEBUG tts] Subprocess exited with code={proc.returncode}")

    if stdout:
        _debug(f"[DEBUG tts] Subprocess stdout: {stdout.decode()[:500]}")
    if stderr:
        _debug(f"[DEBUG tts] Subprocess stderr: {stderr.decode()[:500]}")

    if proc.returncode != 0:
        err_msg = (
            stderr.decode().strip()
            or f"Kokoro subprocess exited with code {proc.returncode}"
        )
        _debug(f"[DEBUG tts] Subprocess failed: {err_msg}")
        await event_bus.emit(
            PipelineError(stage="tts", fatal=False, message=err_msg)
        )
        return False

    _debug("[DEBUG tts] Subprocess succeeded")
    return True


def _build_kokoro_script(voice: str, speed: float = 1.0) -> str:
    """Build the inline Python script the subprocess will run."""
    # Language code derived from voice prefix, defaulting to American English.
    # Kokoro voice names often start with af_ (American Female), am_
    # (American Male), bf_ (British Female), bm_ (British Male), etc.
    lang_code = "a"  # default: American English

    return f'''\
import os, sys, tempfile, subprocess as _sp
from pathlib import Path

# Debug helper — only prints when parent process passed --debug
def _debug(msg):
    if os.environ.get("WHISPER_BOT_DEBUG"):
        print(msg, file=sys.stderr, flush=True)

text = sys.stdin.read()
_debug(f"[DEBUG tts subproc] Received text: {{len(text)}} chars")
if not text.strip():
    _debug("[DEBUG tts subproc] Text is empty/whitespace, exiting")
    sys.exit(0)

# ---- Kokoro initialisation ------------------------------------------------
try:
    from kokoro import KPipeline
    import numpy as np
    _debug("[DEBUG tts subproc] KPipeline and numpy imported successfully")
except ImportError as _exc:
    print(str(_exc), file=sys.stderr)
    sys.exit(1)

try:
    _debug(f"[DEBUG tts subproc] Initialising KPipeline(lang_code={lang_code!r})")
    pipeline = KPipeline(lang_code="{lang_code}")
    _debug("[DEBUG tts subproc] KPipeline initialised")
except Exception as _exc:
    print(f"KPipeline init failed: {{_exc}}", file=sys.stderr)
    sys.exit(1)

# ---- Generate audio -------------------------------------------------------
try:
    _debug(f"[DEBUG tts subproc] Generating audio with voice={voice!r}")
    audio_iter = pipeline(text, voice="{voice}")
    _debug("[DEBUG tts subproc] Audio generator created")
except Exception as _exc:
    print(f"TTS generation failed: {{_exc}}", file=sys.stderr)
    sys.exit(1)

audio_parts: list[np.ndarray] = []
_chunk_count = 0
for result in audio_iter:
    graphemes = result.graphemes
    phonemes = result.phonemes
    audio_out = result.audio  # torch.FloatTensor or None
    if audio_out is None:
        _debug(f"[DEBUG tts subproc] Chunk #{{_chunk_count}}: no audio output, skipping")
        continue
    _chunk_count += 1
    # Convert torch tensor to numpy array for np.concatenate / sounddevice / soundfile
    audio_np: np.ndarray = audio_out.cpu().numpy() if hasattr(audio_out, "cpu") else np.asarray(audio_out)
    audio_parts.append(audio_np)
    _debug(f"[DEBUG tts subproc] Generated audio chunk #{{_chunk_count}}: {{len(audio_np)}} samples (graphemes={{len(graphemes)}} chars, phonemes={{len(phonemes)}} chars)")

_debug(f"[DEBUG tts subproc] Generated {{_chunk_count}} audio chunks total")

if not audio_parts:
    _debug("[DEBUG tts subproc] No audio generated, exiting")
    sys.exit(0)

full_audio = (
    np.concatenate(audio_parts) if len(audio_parts) > 1 else audio_parts[0]
)
_debug(f"[DEBUG tts subproc] Full audio: {{len(full_audio)}} samples")
sr = 24000  # Kokoro default sample rate

# ---- Speed change (resample) ----------------------------------------------
_speed = {speed}
if _speed != 1.0 and len(full_audio) > 0:
    _n_orig = len(full_audio)
    _n_new = max(1, int(_n_orig / _speed))
    _indices = np.linspace(0, _n_orig - 1, _n_new)
    _floor = np.floor(_indices).astype(int)
    _ceil = np.minimum(_floor + 1, _n_orig - 1)
    _frac = _indices - _floor
    full_audio = full_audio[_floor] * (1.0 - _frac) + full_audio[_ceil] * _frac
    _debug(f"[DEBUG tts subproc] Resampled to {{len(full_audio)}} samples (speed={{_speed}})")

# ---- Playback -------------------------------------------------------------
# Prefer sounddevice (direct), fall back to soundfile + afplay (macOS).
try:
    import sounddevice as _sd
    _debug("[DEBUG tts subproc] Using sounddevice for playback")
    _sd.play(full_audio, sr)
    _debug("[DEBUG tts subproc] Playing audio...")
    _sd.wait()
    _debug("[DEBUG tts subproc] Playback complete")
except ImportError:
    _debug("[DEBUG tts subproc] sounddevice not available, trying soundfile+afplay")
    try:
        import soundfile as _sf
        _debug("[DEBUG tts subproc] soundfile imported successfully")
    except ImportError as _exc:
        print(f"Neither sounddevice nor soundfile available: {{_exc}}", file=sys.stderr)
        sys.exit(1)

    _tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        _debug(f"[DEBUG tts subproc] Writing WAV to {{_tmp.name}}")
        _sf.write(_tmp.name, full_audio, sr)
        _debug(f"[DEBUG tts subproc] Playing with afplay...")
        _sp.run(["afplay", _tmp.name], check=True)
        _debug("[DEBUG tts subproc] afplay complete")
    except FileNotFoundError:
        print("afplay not found (not on macOS?)", file=sys.stderr)
        sys.exit(1)
    except Exception as _exc:
        print(f"Playback failed: {{_exc}}", file=sys.stderr)
        sys.exit(1)
    finally:
        Path(_tmp.name).unlink(missing_ok=True)
'''