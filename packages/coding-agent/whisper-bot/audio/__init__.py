"""Audio feedback — non-verbal sounds for pipeline state transitions.

Bundles small .wav files and plays them via ``afplay`` (macOS).
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from whisper_bot.debug import log as _debug


class AudioFeedback:
    """Plays earcons at pipeline state transitions.

    Config keys read from ``config.get("audio_feedback", {})``:

    * ``enabled`` — master switch (default ``True``).
    """

    def __init__(self, config: dict) -> None:
        cfg = config.get("audio_feedback", {})
        self._enabled = cfg.get("enabled", True)
        self._sounds_dir = Path(__file__).parent

        # Sound file names to play
        self._sounds = {
            "start": "start.wav",
            "stop": "stop.wav",
            "chime": "chime.wav",
            "error": "error.wav",
        }

        _debug(f"[DEBUG audio] AudioFeedback initialized: enabled={self._enabled}")

    async def play(self, name: str) -> None:
        """Play a named earcon.  Non-blocking (fire-and-forget subprocess).

        *name* must be one of ``"start"``, ``"stop"``, ``"chime"``, ``"error"``.
        Silently returns if audio feedback is disabled or the sound file
        is missing.
        """
        if not self._enabled or name not in self._sounds:
            return

        path = self._sounds_dir / self._sounds[name]
        if not path.exists():
            _debug(f"[DEBUG audio] Sound file not found: {path}")
            return

        try:
            proc = await asyncio.create_subprocess_exec(
                "afplay",
                str(path),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            _debug(f"[DEBUG audio] Playing {name} (pid={proc.pid})")
            # Fire-and-forget — don't await.  The earcon plays in the
            # background and exits on its own (~100-200 ms later).
        except FileNotFoundError:
            _debug("[DEBUG audio] afplay not found — no audio feedback")
        except Exception as exc:
            _debug(f"[DEBUG audio] Failed to play {name}: {exc}")
