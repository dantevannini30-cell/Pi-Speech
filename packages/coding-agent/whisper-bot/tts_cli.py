#!/usr/bin/env python3
"""
One-shot TTS CLI — speaks text and exits.

Usage:
    python3 tts_cli.py --text "Hello world" [--voice en_US-lessac-medium]

Calls the existing whisper_bot TTS providers directly.
The pi-mono directory is added to sys.path so whisper_bot is importable.
"""

import sys
import os

# Add the pi-mono coding-agent package dir so whisper_bot is importable
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODING_AGENT_DIR = os.path.dirname(_SCRIPT_DIR)  # packages/coding-agent
if _CODING_AGENT_DIR not in sys.path:
    sys.path.insert(0, _CODING_AGENT_DIR)

import argparse
import asyncio
from whisper_bot.config import load_config
from whisper_bot.tts.piper import PiperTTS


async def main() -> None:
    parser = argparse.ArgumentParser(description="One-shot TTS")
    parser.add_argument("--text", required=True, help="Text to speak")
    parser.add_argument("--voice", default=None, help="TTS voice (e.g. en_US-lessac-medium)")
    parser.add_argument("--provider", default="piper", choices=["piper", "kokoro"], help="TTS provider")
    args = parser.parse_args()

    if not args.text.strip():
        return

    config = load_config()
    if args.voice:
        config["tts"]["voice"] = args.voice
    config["tts"]["provider"] = args.provider
    config["tts"]["streaming"] = False  # one-shot mode

    if args.provider == "piper":
        from whisper_bot.tts.piper import PiperTTS
        tts = PiperTTS(config, None)  # no event bus needed for standalone
    else:
        from whisper_bot.tts.kokoro import KokoroTTS
        tts = KokoroTTS(config, None)

    await tts.speak(args.text)


if __name__ == "__main__":
    asyncio.run(main())
