"""Parser provider interface.

Parsers clean up raw STT output — fixing transcription errors, resolving
self-corrections, and normalizing punctuation — before passing the text
to the agent.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ParserProvider(ABC):
    """Interface for text-cleaning parsers."""

    @abstractmethod
    async def parse(self, raw_text: str) -> str:
        """Clean *raw_text* and return the parsed version."""
        ...
