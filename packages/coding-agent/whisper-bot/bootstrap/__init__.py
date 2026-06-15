"""Dependency detection and first-run setup.

Checks for all required external services and prints a consolidated message
for anything missing.
"""

from __future__ import annotations
from whisper_bot.debug import log as _debug

import json
import os
import pathlib
import shutil
import subprocess
import sys
from typing import Any

import httpx

# ---------------------------------------------------------------------------
# Repo-relative paths
# ---------------------------------------------------------------------------

_THIS_DIR = pathlib.Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
_TYPEWHISPER_CHECKOUT = _PROJECT_ROOT / "typewhisper-mac"


def _check_typewhisper_checkout() -> bool:
    """Check if the ``typewhisper-mac/`` checkout exists next to whisper-bot."""
    result = _TYPEWHISPER_CHECKOUT.is_dir()
    _debug(f"[DEBUG bootstrap] typewhisper_checkout: {result} (path={_TYPEWHISPER_CHECKOUT})")
    return result


# ---------------------------------------------------------------------------
# Check helpers
# ---------------------------------------------------------------------------


def _check_pi() -> bool:
    """Check that the ``pi`` CLI is available."""
    pi_path = shutil.which("pi")
    _debug(f"[DEBUG bootstrap] pi CLI: {pi_path!r}")
    return pi_path is not None


def _check_espeak_ng() -> bool:
    """Check that ``espeak-ng`` is installed (required by Kokoro on macOS)."""
    es_path = shutil.which("espeak-ng")
    _debug(f"[DEBUG bootstrap] espeak-ng: {es_path!r}")
    return es_path is not None


def _check_piper() -> bool:
    """Check that the ``piper-tts`` Python package is installed."""
    try:
        import piper  # noqa: F401
        _debug("[DEBUG bootstrap] piper-tts Python package: found")
        return True
    except ImportError:
        _debug("[DEBUG bootstrap] piper-tts Python package: NOT found")
        return False


def _check_piper_model() -> bool:
    """Check that the Piper voice model ``.onnx`` file exists."""
    model_path = os.path.expanduser("~/.whisper-bot/piper/en_US-lessac-medium.onnx")
    exists = os.path.isfile(model_path)
    _debug(f"[DEBUG bootstrap] piper model file: {model_path} exists={exists}")
    return exists


def _check_piper_model_json() -> bool:
    """Check that the Piper voice model ``.onnx.json`` config file exists."""
    json_path = os.path.expanduser("~/.whisper-bot/piper/en_US-lessac-medium.onnx.json")
    json_ok = os.path.isfile(json_path)
    _debug(f"[DEBUG bootstrap] piper model json: {json_path} exists={json_ok}")
    return json_ok


def _check_ollama_running() -> bool:
    """Check if the ollama server is reachable."""
    try:
        _debug("[DEBUG bootstrap] Checking ollama at http://localhost:11434/api/tags...")
        r = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
        ok = r.status_code < 500
        _debug(f"[DEBUG bootstrap] ollama API: status={r.status_code} ok={ok}")
        return ok
    except httpx.ConnectError as e:
        _debug(f"[DEBUG bootstrap] ollama API: ConnectError - {e}")
        return False
    except httpx.TimeoutException as e:
        _debug(f"[DEBUG bootstrap] ollama API: Timeout - {e}")
        return False
    except Exception as e:
        _debug(f"[DEBUG bootstrap] ollama API: Unexpected error - {type(e).__name__}: {e}")
        return False


def _check_ollama_model(model: str = "qwen2.5-coder:3b") -> bool:
    """Check if a specific model is available in ollama."""
    try:
        _debug(f"[DEBUG bootstrap] Checking if model {model!r} is available in ollama...")
        r = httpx.get("http://localhost:11434/api/tags", timeout=3.0)
        if r.status_code != 200:
            _debug(f"[DEBUG bootstrap] ollama API returned status {r.status_code}")
            return False
        models = r.json().get("models", [])
        names = set()
        for m in models:
            if isinstance(m, dict):
                name = m.get("name", "").split(":")[0]
                names.add(name)
                _debug(f"[DEBUG bootstrap]   available model: {name}")
            elif isinstance(m, str):
                names.add(m.split(":")[0])
                _debug(f"[DEBUG bootstrap]   available model: {m.split(':')[0]}")
        found = model in names
        _debug(f"[DEBUG bootstrap] Model {model!r} found: {found}")
        return found
    except (httpx.RequestError, Exception) as e:
        _debug(f"[DEBUG bootstrap] ollama model check failed: {type(e).__name__}: {e}")
        return False


def _find_typewhisper_discovery() -> str | None:
    """Find the TypeWhisper ``api-discovery.json`` path, or return ``None``.

    Dev builds (local checkout via Xcode) are checked first so they take
    precedence over a production install.
    """
    candidates = [
        # Dev build (local checkout — built via Xcode, runs without signing)
        pathlib.Path.home()
        / "Library"
        / "Application Support"
        / "TypeWhisper-Dev"
        / "api-discovery.json",
        # Production build (installed .app)
        pathlib.Path.home()
        / "Library"
        / "Application Support"
        / "TypeWhisper"
        / "api-discovery.json",
    ]
    for path in candidates:
        _debug(f"[DEBUG bootstrap] Checking discovery path: {path}")
        if path.exists():
            _debug(f"[DEBUG bootstrap] Found discovery file at: {path}")
            return str(path)
    _debug("[DEBUG bootstrap] No TypeWhisper discovery file found")
    return None


def _check_typewhisper_running() -> bool:
    """Check if TypeWhisper's HTTP API is reachable."""
    discovery = _find_typewhisper_discovery()
    if not discovery:
        _debug("[DEBUG bootstrap] No discovery file, TypeWhisper not running")
        return False
    try:
        with open(discovery) as f:
            data = json.load(f)
        port = data.get("port", 8978)
        token = data.get("token", "")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        _debug(f"[DEBUG bootstrap] Checking TypeWhisper at 127.0.0.1:{port}/v1/status...")
        r = httpx.get(f"http://127.0.0.1:{port}/v1/status", headers=headers, timeout=3.0)
        ok = r.status_code < 500
        _debug(f"[DEBUG bootstrap] TypeWhisper status: {r.status_code} ok={ok}")
        return ok
    except httpx.ConnectError as e:
        _debug(f"[DEBUG bootstrap] TypeWhisper API: ConnectError - {e}")
        return False
    except httpx.TimeoutException as e:
        _debug(f"[DEBUG bootstrap] TypeWhisper API: Timeout - {e}")
        return False
    except (httpx.RequestError, json.JSONDecodeError, Exception) as e:
        _debug(f"[DEBUG bootstrap] TypeWhisper API: Unexpected error - {type(e).__name__}: {e}")
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_dependencies() -> dict[str, Any]:
    """Run all dependency checks and return a dict of results.

    Return value::

        {
            "pi": bool,
            "espeak_ng": bool,
            "piper": bool,
            "piper_model": bool,
            "piper_model_json": bool,
            "ollama_running": bool,
            "ollama_model": bool | None,   # None if ollama not running
            "typewhisper_running": bool,
            "typewhisper_checkout": bool,  # local typewhisper-mac/ checkout present
        }
    """
    _debug("[DEBUG bootstrap] Running dependency checks...")
    pi_ok = _check_pi()
    espeak_ok = _check_espeak_ng()
    piper_ok = _check_piper()
    piper_model_ok = _check_piper_model()
    piper_model_json_ok = _check_piper_model_json()
    ollama_running = _check_ollama_running()
    ollama_model = _check_ollama_model() if ollama_running else None
    tw_running = _check_typewhisper_running()
    tw_checkout = _check_typewhisper_checkout()

    return {
        "pi": pi_ok,
        "espeak_ng": espeak_ok,
        "piper": piper_ok,
        "piper_model": piper_model_ok,
        "piper_model_json": piper_model_json_ok,
        "ollama_running": ollama_running,
        "ollama_model": ollama_model,
        "typewhisper_running": tw_running,
        "typewhisper_checkout": tw_checkout,
    }


def print_status(results: dict[str, Any]) -> None:
    """Print a human-readable status summary to stderr."""
    lines: list[str] = []

    if not results["typewhisper_running"]:
        if results.get("typewhisper_checkout"):
            lines.append(
                "  TypeWhisper is not running.  Build & run from the local checkout:\n"
                "    cd typewhisper-mac && open TypeWhisper.xcodeproj\n"
                "  Then Cmd+R to run.  Enable Settings > Advanced > API Server.\n"
            )
        else:
            lines.append(
                "  TypeWhisper is not running.  Install it:\n"
                "    brew install --cask typewhisper/tap/typewhisper\n"
                "  Or download from https://typewhisper.com\n"
            )

    if not results["pi"]:
        lines.append(
            "  Pi agent CLI not found.  Install it:\n"
            "    npm install -g @earendil-works/pi-coding-agent\n"
        )

    if not results["ollama_running"]:
        lines.append(
            "  Ollama is not running.  Install & start it:\n"
            "    brew install ollama && ollama serve\n"
        )
    elif not results.get("ollama_model"):
        lines.append(
            "  Qwen2.5-Coder 3B model not found in ollama.  Pull it:\n"
            "    ollama pull qwen2.5-coder:3b\n"
        )

    if not results["espeak_ng"]:
        lines.append(
            "  espeak-ng not found (required by Kokoro TTS).  Install it:\n"
            "    brew install espeak-ng\n"
        )

    if not results.get("piper"):
        lines.append(
            "  Piper TTS package not found (default TTS engine).  Install it:\n"
            "    pip install piper-tts\n"
            "  Then download a voice model:\n"
            "    mkdir -p ~/.whisper-bot/piper\n"
            "    curl -L https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx \\\n"
            "      -o ~/.whisper-bot/piper/en_US-lessac-medium.onnx\n"
            "    curl -L https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json \\\n"
            "      -o ~/.whisper-bot/piper/en_US-lessac-medium.onnx.json\n"
        )
    else:
        if not results.get("piper_model"):
            lines.append(
                "  Piper voice model file not found.  Download it:\n"
                "    curl -L https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx \\\n"
                "      -o ~/.whisper-bot/piper/en_US-lessac-medium.onnx\n"
            )
        if not results.get("piper_model_json"):
            lines.append(
                "  Piper voice model config file not found.  Download it:\n"
                "    curl -L https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json \\\n"
                "      -o ~/.whisper-bot/piper/en_US-lessac-medium.onnx.json\n"
            )

    if lines:
        print("whisper-bot dependency check:", file=sys.stderr)
        for line in lines:
            print(line, file=sys.stderr)
    else:
        print("whisper-bot: all dependencies found.", file=sys.stderr)
