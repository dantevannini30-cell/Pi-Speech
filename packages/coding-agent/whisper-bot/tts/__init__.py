"""Text-to-Speech provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod


class TTSProvider(ABC):
    """Interface for text-to-speech providers."""

    @abstractmethod
    async def speak(self, text: str) -> None:
        """Speak *text* aloud (non-streaming)."""
        ...
