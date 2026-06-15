"""TypeWhisper STT provider — communicates with the TypeWhisper API via HTTP."""

from __future__ import annotations
from whisper_bot.debug import log as _debug

import asyncio
import json
import os
import sys
from pathlib import Path

import httpx

from whisper_bot.events.bus import EventBus
from whisper_bot.events.types import (
    PipelineError,
    STTDone,
    STTStarted,
    STTTranscribing,
)
from whisper_bot.stt import STTProvider

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

_DISCOVERY_PATHS = [
    # Dev build (built from local checkout via Xcode)
    Path("~/Library/Application Support/TypeWhisper-Dev/api-discovery.json").expanduser(),
    # Production build (installed .app)
    Path("~/Library/Application Support/TypeWhisper/api-discovery.json").expanduser(),
]
_DEFAULT_PORT = 8978


def _discover() -> tuple[int, str | None]:
    """Return ``(port, token)``, trying discovery files in priority order.

    Dev builds (``TypeWhisper-Dev``) are checked first so a local checkout
    takes precedence over a production install.
    """
    for path in _DISCOVERY_PATHS:
        _debug(f"[DEBUG stt] Checking discovery path: {path}")
        if path.exists():
            try:
                data = json.loads(path.read_text())
                port = int(data["port"])
                token = data.get("token")
                _debug(f"[DEBUG stt] Found discovery at {path}: port={port}, token={'<set>' if token else 'None'}")
                return port, token
            except (KeyError, ValueError, OSError) as e:
                _debug(f"[DEBUG stt] Failed to parse {path}: {e}")
                continue
    _debug(f"[DEBUG stt] No discovery file found, using defaults: port={_DEFAULT_PORT}, token=None")
    return _DEFAULT_PORT, None


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class TypeWhisperProvider(STTProvider):
    """Speech-to-text provider backed by the TypeWhisper local API."""

    def __init__(self, config: dict, event_bus: EventBus) -> None:
        self.config = config
        self.event_bus = event_bus
        self._session_id: str | None = None

        port, token = _discover()
        self._port = port
        self._token = token

        stt_config = config.get("stt", {})
        self._engine = stt_config.get("engine", "parakeet")

        # Override from config if present
        if stt_config.get("port") is not None:
            self._port = int(stt_config["port"])
            _debug(f"[DEBUG stt] Port overridden by config: {self._port}")
        if stt_config.get("api_token") is not None:
            self._token = stt_config["api_token"]
            _debug(f"[DEBUG stt] Token overridden by config")

        self._client = httpx.AsyncClient(timeout=30.0)

        _debug(f"[DEBUG stt] TypeWhisperProvider initialized: base_url={self._base_url()} engine={self._engine} token={'<set>' if self._token else 'None'}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _base_url(self) -> str:
        return f"http://127.0.0.1:{self._port}"

    def _headers(self) -> dict[str, str]:
        headers = {"x-engine": self._engine}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    # ------------------------------------------------------------------
    # Health / auto-launch
    # ------------------------------------------------------------------

    async def _ensure_running(self) -> bool:
        """Ensure TypeWhisper is running. Try to launch it if not found."""
        status_url = f"{self._base_url()}/v1/status"

        # 1. Quick check — is the API already reachable?
        _debug(f"[DEBUG stt] Checking if TypeWhisper is already running at {status_url}")
        try:
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(status_url, headers=self._headers())
                if resp.status_code == 200:
                    _debug("[DEBUG stt] TypeWhisper is already running")
                    return True
        except (httpx.RequestError, httpx.HTTPStatusError):
            pass

        # 2. Not running — try to launch
        _debug("[DEBUG stt] TypeWhisper not running, launching...")
        proc = await asyncio.create_subprocess_exec(
            "open", "/Applications/TypeWhisper.app",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        _debug(f"[DEBUG stt] Launch command exited with code {proc.returncode}")

        # 3. Poll for up to 30 s (every 500 ms)
        deadline = asyncio.get_event_loop().time() + 30.0
        poll_count = 0
        _debug("[DEBUG stt] Polling for TypeWhisper to become ready...")
        while asyncio.get_event_loop().time() < deadline:
            poll_count += 1
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    resp = await client.get(status_url, headers=self._headers())
                    if resp.status_code == 200:
                        _debug(f"[DEBUG stt] TypeWhisper ready after ~{poll_count * 0.5:.1f}s")
                        return True
            except (httpx.RequestError, httpx.HTTPStatusError):
                pass
            await asyncio.sleep(0.5)

        _debug(f"[DEBUG stt] TypeWhisper failed to start after {poll_count} polls ({30.0}s)")
        return False

    # ------------------------------------------------------------------
    # STTProvider interface
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Begin a dictation session with TypeWhisper."""
        url = f"{self._base_url()}/v1/dictation/start"
        _debug(f"[DEBUG stt] POST {url} headers={self._headers()}")
        try:
            resp = await self._client.post(url, headers=self._headers())
            _debug(f"[DEBUG stt] POST /start response: {resp.status_code} {resp.text[:200]}")
            resp.raise_for_status()
        except httpx.RequestError as exc:
            _debug(f"[DEBUG stt] RequestError on /start: {exc}")
            raise PipelineError(
                stage="stt",
                message=f"Failed to start dictation: {exc}",
                fatal=True,
            ) from exc
        except httpx.HTTPStatusError as exc:
            _debug(f"[DEBUG stt] HTTPStatusError on /start: {exc}")
            raise PipelineError(
                stage="stt",
                message=f"Failed to start dictation: {exc}",
                fatal=True,
            ) from exc

        data = resp.json()
        self._session_id = data["id"]
        _debug(f"[DEBUG stt] Dictation session started: id={self._session_id}")
        await self.event_bus.emit(STTStarted())

    async def stop(self) -> str:
        """Stop dictation and return the transcribed text.

        Sends the stop request, then polls for the transcription result
        every 500 ms until it is available or 60 s elapses.
        """
        if self._session_id is None:
            _debug("[DEBUG stt] stop() called but _session_id is None!")
            raise PipelineError(
                stage="stt",
                message="No active dictation session to stop.",
                fatal=True,
            )

        url = f"{self._base_url()}/v1/dictation/stop"
        _debug(f"[DEBUG stt] POST {url} session_id={self._session_id}")
        try:
            resp = await self._client.post(
                url,
                headers=self._headers(),
                json={"id": self._session_id},
            )
            _debug(f"[DEBUG stt] POST /stop response: {resp.status_code} {resp.text[:200]}")
            resp.raise_for_status()
        except httpx.RequestError as exc:
            _debug(f"[DEBUG stt] RequestError on /stop: {exc}")
            raise PipelineError(
                stage="stt",
                message=f"Failed to stop dictation: {exc}",
                fatal=True,
            ) from exc
        except httpx.HTTPStatusError as exc:
            _debug(f"[DEBUG stt] HTTPStatusError on /stop: {exc}")
            raise PipelineError(
                stage="stt",
                message=f"Failed to stop dictation: {exc}",
                fatal=True,
            ) from exc

        await self.event_bus.emit(STTTranscribing())

        # Poll for transcription
        poll_url = f"{self._base_url()}/v1/dictation/transcription"
        params = {"id": self._session_id}
        deadline = asyncio.get_event_loop().time() + 60.0
        poll_count = 0

        _debug(f"[DEBUG stt] Polling {poll_url} for transcription...")
        text = ""
        while asyncio.get_event_loop().time() < deadline:
            poll_count += 1
            try:
                poll_resp = await self._client.get(
                    poll_url,
                    headers=self._headers(),
                    params=params,
                )
                _debug(f"[DEBUG stt] Poll #{poll_count}: status={poll_resp.status_code} text={poll_resp.text[:200]}")
                poll_resp.raise_for_status()
                data = poll_resp.json()
                text = data.get("text") or data.get("transcription", {}).get("text", "")
                if text:
                    _debug(f"[DEBUG stt] Transcription received on poll #{poll_count}: {text!r} (from data.transcription.text)")
                    break
                _debug(f"[DEBUG stt] Poll #{poll_count}: text empty, sleeping 0.5s...")
            except httpx.RequestError as exc:
                _debug(f"[DEBUG stt] RequestError during poll #{poll_count}: {exc}")
                raise PipelineError(
                    stage="stt",
                    message=f"Transcription polling failed: {exc}",
                    fatal=True,
                ) from exc
            except httpx.HTTPStatusError as exc:
                _debug(f"[DEBUG stt] HTTPStatusError during poll #{poll_count}: {exc}")
                raise PipelineError(
                    stage="stt",
                    message=f"Transcription polling failed: {exc}",
                    fatal=True,
                ) from exc

            await asyncio.sleep(0.5)
        else:
            _debug(f"[DEBUG stt] TIMEOUT after 60s and {poll_count} polls! Last text: {text!r}")
            raise PipelineError(
                stage="stt",
                message="Transcription timed out after 60 s.",
                fatal=True,
            )

        _debug(f"[DEBUG stt] Emitting STTDone with text: {text!r}")
        await self.event_bus.emit(STTDone(text=text))
        return text