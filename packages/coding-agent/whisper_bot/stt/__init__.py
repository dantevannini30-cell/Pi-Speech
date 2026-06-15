"""Speech-to-Text provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class STTProvider(ABC):
    """Interface for speech-to-text providers.

    Implementations wrap services like TypeWhisper, Whisper.cpp, etc.
    """

    @abstractmethod
    async def start(self) -> None:
        """Begin recording audio for transcription."""
        ...

    @abstractmethod
    async def stop(self) -> str:
        """Stop recording and return the transcribed text."""
        ...
