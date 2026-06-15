"""Ollama-backed parser provider for whisper-bot.

Uses the OpenAI-compatible chat completions endpoint exposed by Ollama
to clean up raw STT output.
"""

from __future__ import annotations
from whisper_bot.debug import log as _debug

import sys
import re
from typing import TYPE_CHECKING

import httpx

from whisper_bot.parser import ParserProvider
from whisper_bot.events.types import ParsingDone, ParsingStarted, PipelineError

if TYPE_CHECKING:
    from whisper_bot.events.bus import EventBus

SYSTEM_PROMPT = (
    "You are a speech-to-text cleanup tool. Your ONLY job is to clean up "
    "transcribed speech. You are NOT a chatbot — do NOT respond, answer, "
    "greet, or react to the content.\n"
    "\n"
    "CRITICAL: You are cleaning TRANSCRIBED SPEECH, not generating a "
    "response. Every word you output must come from the original speech "
    "(minus fillers and corrected parts). Never add information, never "
    "rephrase as a response, and NEVER generate code, markdown, or code "
    "blocks. If the user asked a question, output the question — not the "
    "answer. If the user gave a command, output the command — not its "
    "result.\n"
    "\n"
    "Rules:\n"
    "1. Remove filler words (um, uh, you know, I mean) — but 'like' as a "
    "qualifier (e.g. 'that file, like the one in docs') is acceptable to "
    "keep.\n"
    "2. Resolve self-corrections: when the speaker changes their mind, "
    "drop the rejected part and keep only the final intent.\n"
    "   - 'wait no I mean X' → X\n"
    "   - 'actually no X' → drop X\n"
    "   - 'forget that / ignore that / scratch that, just Y' → Y\n"
    "   - 'this is a test, actually no forget that just Y' → Y\n"
    "3. Fix obvious transcription errors (homophones like "
    "their/they're/there, its/it's, to/too/two, your/you're, "
    "then/than; run-together words like 'wassup' → 'what's up'; "
    "dot → . in file names). IMPORTANT: check the FIRST word for "
    "homophones too — 'their' at sentence start becomes 'They're' "
    "(not 'Their'), 'your' becomes 'You're' (not 'Your').\n"
    "4. Normalize punctuation and sentence casing\n"
    "5. Preserve ALL original meaning and intent. Keep the speech form "
    "(question, command, request) exactly as spoken. Never summarize or "
    "turn a request into its result.\n"
    "\n"
    "Examples:\n"
    'Input: "um can you um actually no wait can you say hi thanks"\n'
    'Output: "Can you say hi? Thanks."\n'
    '\n'
    'Input: "what\'s the weather like in um I mean what time is it"\n'
    'Output: "What time is it?"\n'
    '\n'
    'Input: "uh yeah so like I need to um find that file"\n'
    'Output: "I need to find that file."\n'
    '\n'
    'Input: "this is a test, actually no forget that just say hi"\n'
    'Output: "Just say hi."\n'
    '\n'
    'Input: "open the um scratch that close the window"\n'
    'Output: "Close the window."\n'
    '\n'
    'Input: "write a file, actually no respond and say hi"\n'
    'Output: "Respond and say hi."\n'
    '\n'
    'Input: "hey um can you create a python script called test dot py '
    'that prints hello world and then um run it"\n'
    'Output: "Can you create a Python script called test.py that prints '
    'hello world and then run it?"\n'
    '\n'
    'Input: "their going to the store and its raining"\n'
    'Output: "They\'re going to the store and it\'s raining."\n'
    "\n"
    "Output ONLY the cleaned text. Never add greetings, explanations, "
    "code, markdown, or responses."
)


class OllamaParser(ParserProvider):
    """Parser that delegates cleanup to an Ollama model via its OpenAI-compatible API."""

    def __init__(self, config: dict, event_bus: EventBus) -> None:
        parser_cfg = config.get("parser", {})
        self.endpoint: str = parser_cfg.get("endpoint", "http://localhost:11434/v1")
        self.model: str = parser_cfg.get("model", "qwen2.5-coder:3b")
        self.timeout: float = parser_cfg.get("timeout_seconds", 5)
        self._event_bus = event_bus
        _debug(f"[DEBUG parser] OllamaParser initialized: endpoint={self.endpoint} model={self.model} timeout={self.timeout}s")

    async def parse(self, raw_text: str) -> str:
        """Clean *raw_text* via the configured Ollama model."""
        _debug(f"[DEBUG parser] parse() called with text ({len(raw_text)} chars): {raw_text!r}")
        await self._event_bus.emit(ParsingStarted())

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": raw_text},
            ],
            "temperature": 0.1,
        }

        _debug(f"[DEBUG parser] Sending POST to {self.endpoint}/chat/completions with model={self.model}")
        _debug(f"[DEBUG parser] Request payload: {payload}")

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(
                    f"{self.endpoint}/chat/completions",
                    json=payload,
                )
                _debug(f"[DEBUG parser] Response status: {resp.status_code}")
                resp.raise_for_status()
                data = resp.json()
                _debug(f"[DEBUG parser] Response JSON: {data}")
                cleaned: str = data["choices"][0]["message"]["content"]
                cleaned = self._post_process(cleaned)
                _debug(f"[DEBUG parser] Cleaned text ({len(cleaned)} chars): {cleaned!r}")
        except httpx.TimeoutException as exc:
            _debug(f"[DEBUG parser] TIMEOUT after {self.timeout}s: {exc}")
            await self._event_bus.emit(
                PipelineError(
                    stage="parser",
                    fatal=False,
                    message=f"Ollama API timed out after {self.timeout}s: {exc}. Falling back to raw text.",
                )
            )
            return raw_text
        except httpx.HTTPError as exc:
            _debug(f"[DEBUG parser] HTTP error: {exc}")
            await self._event_bus.emit(
                PipelineError(
                    stage="parser",
                    fatal=False,
                    message=f"Ollama API error: {exc}. Falling back to raw text.",
                )
            )
            return raw_text
        except (KeyError, ValueError) as exc:
            _debug(f"[DEBUG parser] Parse/response error: {exc}")
            await self._event_bus.emit(
                PipelineError(
                    stage="parser",
                    fatal=False,
                    message=f"Ollama API error: {exc}. Falling back to raw text.",
                )
            )
            return raw_text

        await self._event_bus.emit(ParsingDone(original=raw_text, cleaned=cleaned))
        _debug(f"[DEBUG parser] ParsingDone event emitted, returning cleaned text")
        return cleaned

    # Words that indicate a verb/contraction context (not a possessive).
    _THEYRE_TRIGGERS: tuple[str, ...] = (
        "going", "coming", "running", "trying", "doing", "making",
        "taking", "getting", "saying", "asking", "telling", "working",
        "all", "not", "really", "very", "so", "just", "already",
        "still", "also", "here", "there", "now",
    )
    _YOURE_TRIGGERS: tuple[str, ...] = (
        "going", "coming", "running", "trying", "doing", "making",
        "taking", "getting", "saying", "asking", "telling", "working",
        "a", "an", "the", "so", "very", "really", "not", "just",
        "right", "wrong", "correct", "amazing", "awesome", "great",
        "welcome", "all", "already", "still",
    )

    @classmethod
    def _post_process(cls, text: str) -> str:
        """Apply deterministic fixes for homophone errors the LLM misses.

        Common STT homophone errors that are easy to fix with simple rules.
        The LLM handles most cases but small models can miss these,
        especially first-word homophones where capitalization confuses them.
        Also reverses model mistakes like expanding "their" → "Their are".
        """
        # 1) Undo model-introduced contraction expansions:
        #    "Their are" / "their are" → "They're" (model expands the contraction)
        text = re.sub(
            r"\b(?P<cap>T?)heir are\b",
            r"\g<cap>hey're",
            text,
            flags=re.IGNORECASE,
        )
        #    "Your are" / "your are" → "You're"
        text = re.sub(
            r"\b(?P<cap>Y?)our are\b",
            r"\g<cap>ou're",
            text,
            flags=re.IGNORECASE,
        )

        # 2) "their" + trigger word → "they're" (case-insensitive, preserve case)
        triggers_re = "|".join(cls._THEYRE_TRIGGERS)
        text = re.sub(
            rf"\b(?P<cap>T?)heir\s+(?=({triggers_re})\b)",
            r"\g<cap>hey're ",
            text,
            flags=re.IGNORECASE,
        )
        # 3) "your" + trigger word → "you're" (case-insensitive, preserve case)
        triggers_re = "|".join(cls._YOURE_TRIGGERS)
        text = re.sub(
            rf"\b(?P<cap>Y?)our\s+(?=({triggers_re})\b)",
            r"\g<cap>ou're ",
            text,
            flags=re.IGNORECASE,
        )
        return text