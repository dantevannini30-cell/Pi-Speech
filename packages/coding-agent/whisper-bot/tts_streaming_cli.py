#!/usr/bin/env python3
"""
Streaming TTS CLI — reads sentences from stdin, speaks each immediately.

Usage:
    echo "Hello world. How are you?" | python3 tts_streaming_cli.py
    # or sentences piped one per line

Reads from stdin line by line, speaks each non-empty line via the configured
TTS provider (Piper or Kokoro). Sentences can be split by the caller for
low-latency streaming speech while the agent is still generating.
"""

import sys
import os

# Add the pi-mono coding-agent package dir so whisper_bot is importable
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CODING_AGENT_DIR = os.path.dirname(_SCRIPT_DIR)  # packages/coding-agent
if _CODING_AGENT_DIR not in sys.path:
    sys.path.insert(0, _CODING_AGENT_DIR)

import asyncio
import argparse
from whisper_bot.config import load_config


async def main() -> None:
    parser = argparse.ArgumentParser(description="Streaming TTS — reads sentences from stdin")
    parser.add_argument("--voice", default=None, help="TTS voice (e.g. en_US-lessac-medium)")
    parser.add_argument("--provider", default="piper", choices=["piper", "kokoro"], help="TTS provider")
    parser.add_argument("--pool-size", type=int, default=2, help="Number of parallel TTS workers")
    args = parser.parse_args()

    config = load_config()
    if args.voice:
        config["tts"]["voice"] = args.voice
    config["tts"]["provider"] = args.provider
    config["tts"]["streaming"] = True
    config["tts"]["pool_size"] = args.pool_size

    if args.provider == "piper":
        from whisper_bot.tts.piper import StreamingPiperTTS
        tts = StreamingPiperTTS(config, None)
    else:
        from whisper_bot.tts.streaming_kokoro import StreamingKokoroTTS
        tts = StreamingKokoroTTS(config, None)

    # Warm up the TTS engine (pre-load models, launch workers)
    await tts.warmup()

    try:
        for line in sys.stdin:
            sentence = line.strip()
            if sentence:
                await tts.speak_sentence(sentence)
    finally:
        await tts.wait()


if __name__ == "__main__":
    asyncio.run(main())
