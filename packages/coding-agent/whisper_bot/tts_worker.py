#!/usr/bin/env python3
"""
Persistent TTS worker — keeps PiperVoice loaded and listens for JSONL on stdin.

Protocol (JSONL stdin/stdout):
  -> {"type":"speak","text":"..."}
  <- {"type":"done","text":"..."}       (playback complete)
  <- {"type":"error","message":"..."}   (TTS failed)
  -> {"type":"shutdown"}
  <- {"type":"ready"}                   (sent once at startup)

Usage:
  python3 tts_worker.py [--provider piper] [--voice en_US-lessac-medium]
"""

import sys, os, json, struct

# Add coding-agent to path so whisper_bot is importable
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODING_AGENT_DIR = os.path.dirname(_SCRIPT_DIR)
if _CODING_AGENT_DIR not in sys.path:
    sys.path.insert(0, _CODING_AGENT_DIR)

import argparse
from whisper_bot.config import load_config


def _emit(msg: dict) -> None:
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def _speak_piper(text: str, model_path: str) -> None:
    """Generate and play audio using PiperVoice directly (no CLI subprocess)."""
    import sounddevice as sd
    import numpy as np
    from piper.voice import PiperVoice

    voice = PiperVoice.load(model_path)
    sr = voice.config.sample_rate

    chunks = []
    for chunk in voice.synthesize(text):
        chunks.append(chunk.audio_float_array)

    if not chunks:
        return

    audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]

    # Play via sounddevice
    stream = sd.OutputStream(samplerate=sr, channels=1, blocksize=0)
    stream.start()
    stream.write(audio.astype(np.float32))
    stream.stop()
    stream.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Persistent TTS worker")
    parser.add_argument("--voice", default=None)
    parser.add_argument("--provider", default="piper", choices=["piper", "kokoro", "say"])
    args = parser.parse_args()

    config = load_config()
    if args.voice:
        config["tts"]["voice"] = args.voice
    config["tts"]["provider"] = args.provider

    # Resolve model path
    if args.provider == "piper":
        from whisper_bot.tts.piper import _resolve_model_path
        model_path = _resolve_model_path(config)
    else:
        model_path = None

    # Pre-load PiperVoice
    voice = None
    if args.provider == "piper":
        try:
            from piper.voice import PiperVoice
            voice = PiperVoice.load(model_path)
            _emit({"type": "ready"})
        except Exception as e:
            _emit({"type": "error", "message": f"PiperVoice load failed: {e}"})
            sys.exit(1)
    else:
        _emit({"type": "ready"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        msg_type = msg.get("type")

        if msg_type == "shutdown":
            break

        elif msg_type == "speak":
            text = msg.get("text", "")
            if not text.strip():
                _emit({"type": "done", "text": text})
                continue

            try:
                if args.provider == "piper" and voice is not None:
                    import sounddevice as sd
                    import numpy as np

                    sr = voice.config.sample_rate
                    chunks = []
                    for chunk in voice.synthesize(text):
                        chunks.append(chunk.audio_float_array)

                    if chunks:
                        audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
                        stream = sd.OutputStream(samplerate=sr, channels=1, blocksize=0)
                        stream.start()
                        stream.write(audio.astype(np.float32))
                        stream.stop()
                        stream.close()

                    _emit({"type": "done", "text": text})

                elif args.provider == "kokoro":
                    from whisper_bot.tts.kokoro import KokoroTTS
                    from whisper_bot.events import PipelineError

                    class _Bus:
                        async def emit(self, ev):
                            if isinstance(ev, PipelineError):
                                _emit({"type": "error", "message": ev.message})

                    import asyncio
                    tts = KokoroTTS(config, _Bus())
                    asyncio.run(tts.speak(text))
                    _emit({"type": "done", "text": text})

                else:
                    # Fallback: macOS say
                    import subprocess
                    subprocess.run(["say", text], check=True)
                    _emit({"type": "done", "text": text})

            except Exception as e:
                _emit({"type": "error", "message": str(e)})


if __name__ == "__main__":
    main()
