"""Default configuration values for whisper-bot.

All values have sensible defaults — the config file at ``~/.whisper-bot/config.json``
is optional, not required.  Environment variables prefixed with ``WHISPER_BOT_``
override any value at load time.
"""

from __future__ import annotations

from typing import Any

DEFAULT_CONFIG: dict[str, Any] = {
    "stt": {
        "provider": "typewhisper",
        "engine": "parakeet",
        "port": None,
        "api_token": None,
    },
    "parser": {
        "provider": "ollama",
        "endpoint": "http://localhost:11434/v1",
        "model": "qwen2.5-coder:3b",
        "timeout_seconds": 30,
        "skip": False,  # skip LLM parsing, pass raw STT straight to agent
    },
    "agent": {
        "provider": "pi",
        "mode": "rpc",  # "rpc" (persistent subprocess) | "oneshot" (spawn per turn)
        "working_directory": "~",
        "timeout_seconds": 120,
        "session_dir": "~/.whisper-bot/sessions",
        "resume_session": True,
    },
    "tts": {
        "provider": "piper",
        "voice": "en_US-lessac-medium",
        "enabled": True,
        "streaming": True,           # sentence-by-sentence streaming TTS
        "pool_size": 2,              # parallel subprocess workers for pipelined gen/play
    },
    "audio_feedback": {
        "enabled": True,
    },
    "display": {
        "show_tool_calls": False,
    },
}
