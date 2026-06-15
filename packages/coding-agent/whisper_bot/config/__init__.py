"""Configuration loader for whisper-bot.

Reads ``~/.whisper-bot/config.json``, auto-creates with defaults if absent,
and applies environment-variable overrides (``WHISPER_BOT_*``).
"""

from __future__ import annotations
from whisper_bot.debug import log as _debug

import json
import os
import pathlib
import sys
from copy import deepcopy
from typing import Any

from whisper_bot.config.defaults import DEFAULT_CONFIG

_CONFIG_DIR = pathlib.Path.home() / ".whisper-bot"
_CONFIG_PATH = _CONFIG_DIR / "config.json"

# Env-var override mapping:  WHISPER_BOT_{SECTION}_{KEY}
# e.g. WHISPER_BOT_TTS_VOICE -> config["tts"]["voice"]
_ENV_PREFIX = "WHISPER_BOT_"


def _apply_env_overrides(config: dict[str, Any]) -> None:
    """Mutate *config* in-place by checking ``WHISPER_BOT_*`` env vars."""
    prefix_len = len(_ENV_PREFIX)
    for key, value in sorted(os.environ.items()):
        if not key.startswith(_ENV_PREFIX):
            continue
        rest = key[prefix_len:].lower().split("_", 1)
        if len(rest) != 2:
            continue
        section, setting = rest
        if section not in config or setting not in config[section]:
            _debug(f"[DEBUG config] Env override {key}={value!r}: unknown section/setting '{section}.{setting}', skipping")
            continue

        # Coerce to the type of the existing default value
        existing = config[section][setting]
        if isinstance(existing, bool):
            parsed = value.lower() in ("true", "1", "yes")
        elif isinstance(existing, int):
            parsed = int(value)
        elif isinstance(existing, float):
            parsed = float(value)
        elif existing is None or isinstance(existing, str):
            parsed = value
        else:
            _debug(f"[DEBUG config] Env override {key}={value!r}: unsupported type {type(existing).__name__}, skipping")
            continue

        _debug(f"[DEBUG config] Env override: {section}.{setting} = {parsed!r} (was {existing!r})")
        config[section][setting] = parsed


def load_config() -> dict[str, Any]:
    """Load configuration, auto-creating the file with defaults if missing.

    Returns a deep copy of the merged config (mutating the returned dict will
    not affect future calls).
    """
    _debug(f"[DEBUG config] Loading config from {_CONFIG_PATH}")

    if not _CONFIG_PATH.exists():
        _debug(f"[DEBUG config] Config file not found, creating with defaults at {_CONFIG_PATH}")
        _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(_CONFIG_PATH, "w") as f:
            json.dump(DEFAULT_CONFIG, f, indent=2)
        config = deepcopy(DEFAULT_CONFIG)
    else:
        _debug(f"[DEBUG config] Config file exists, reading...")
        with open(_CONFIG_PATH) as f:
            file_config: dict[str, Any] = json.load(f)
        # Merge: start with defaults, overlay file values
        config = deepcopy(DEFAULT_CONFIG)
        for section, values in file_config.items():
            if section in config and isinstance(values, dict):
                merged = {k: v for k, v in values.items() if k in config[section]}
                if merged:
                    _debug(f"[DEBUG config] Merging {section}: {merged}")
                config[section].update(merged)

    _apply_env_overrides(config)

    _debug(f"[DEBUG config] Final config: stt={config.get('stt')}")
    _debug(f"[DEBUG config] Final config: parser={config.get('parser')}")
    _debug(f"[DEBUG config] Final config: agent={config.get('agent')}")
    _debug(f"[DEBUG config] Final config: tts={config.get('tts')}")
    _debug(f"[DEBUG config] Final config: display={config.get('display')}")

    return config
