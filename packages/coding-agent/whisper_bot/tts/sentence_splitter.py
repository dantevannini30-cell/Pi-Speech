"""Buffers tokens and yields complete sentences for streaming TTS."""

from __future__ import annotations
import re


class SentenceSplitter:
    """Accumulates tokens and splits on sentence boundaries.

    Designed for streaming TTS: feed tokens as they arrive, get back
    complete sentences that can be spoken incrementally.

    Rules:
    - Split on ``.``, ``!``, ``?`` followed by whitespace or end-of-string.
    - Minimum sentence length: 3 words (avoid "Hi." on its own).
    - Max buffer: force-flush after 50 words even without punctuation.
    - Code-block aware: tracks ``` fences and doesn't split inside them.
    """

    def __init__(self, min_words: int = 3, max_buffer_words: int = 50) -> None:
        self._buffer: list[str] = []
        self._word_count = 0
        self._min_words = min_words
        self._max_buffer_words = max_buffer_words
        self._in_code_block = False
        # Split on . ! ? followed by whitespace or end
        self._split_pattern = re.compile(r'([.!?])(?:\s+|$)')

    def feed(self, token: str) -> list[str]:
        """Feed a token, return a list of complete sentences split off.

        The token is appended to an internal buffer.  Whenever the buffer
        ends with a sentence boundary and meets the minimum word count, it
        is split off and returned.
        """
        if not token:
            return []

        # Track code-block fences
        if '```' in token:
            self._in_code_block = not self._in_code_block

        self._buffer.append(token)
        words = token.split()
        self._word_count += len(words)

        # Don't split inside code blocks
        if self._in_code_block:
            return []

        return self._flush_complete_sentences()

    def flush(self) -> list[str]:
        """Return the remaining buffer as a single sentence."""
        if not self._buffer:
            return []
        sentence = ''.join(self._buffer).strip()
        word_count = len(sentence.split())
        self._buffer.clear()
        self._word_count = 0
        if sentence and word_count >= self._min_words:
            return [sentence]
        elif sentence:
            return [sentence]
        return []

    def _flush_complete_sentences(self) -> list[str]:
        """Split complete sentences off the buffer and return them.

        Iterates through every sentence boundary and releases each qualifying
        sentence individually.  This handles both streaming input (one word at
        a time — releases one sentence per boundary) and batch input (full
        text at once — releases multiple short sentences) correctly.
        """
        text = ''.join(self._buffer)
        matches = list(self._split_pattern.finditer(text))
        if not matches:
            # Force-flush if buffer exceeds max words
            if self._word_count >= self._max_buffer_words:
                return self._flush_all()
            return []

        sentences: list[str] = []
        last_end = 0
        for match in matches:
            end = match.end()
            candidate = text[last_end:end].strip()
            word_count = len(candidate.split()) if candidate else 0
            if word_count >= self._min_words:
                sentences.append(candidate)
                last_end = end
            # Fragments below min_words are absorbed into the next sentence

        if not sentences:
            return []

        # Keep remainder in buffer for the next feed
        remainder = text[last_end:]
        self._buffer = [remainder] if remainder else []
        self._word_count = len(remainder.split()) if remainder else 0

        return sentences

    def _flush_all(self) -> list[str]:
        sentence = ''.join(self._buffer).strip()
        self._buffer.clear()
        self._word_count = 0
        return [sentence] if sentence else []
