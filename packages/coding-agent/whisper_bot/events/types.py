"""Typed event dataclasses for the whisper-bot pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union


# ---------------------------------------------------------------------------
# Input events
# ---------------------------------------------------------------------------


@dataclass
class InputCommand:
    """A command entered by the user via stdin."""
    action: str  # "listen" | "stop" | "quit" | "help"


# ---------------------------------------------------------------------------
# STT events
# ---------------------------------------------------------------------------


@dataclass
class STTStarted:
    """Recording has begun."""
    pass


@dataclass
class STTTranscribing:
    """Waiting for TypeWhisper to return transcription (show dots)."""
    pass


@dataclass
class STTDone:
    """Raw transcription received from STT."""
    text: str


# ---------------------------------------------------------------------------
# Parser events
# ---------------------------------------------------------------------------


@dataclass
class ParsingStarted:
    """Parser is cleaning the raw text."""
    pass


@dataclass
class ParsingDone:
    """Parser has finished cleaning the text."""
    original: str  # raw STT text
    cleaned: str   # parsed / cleaned text


# ---------------------------------------------------------------------------
# Agent events
# ---------------------------------------------------------------------------


@dataclass
class UserTextInput:
    """Text entered by the user directly (typed, not from STT).

    Goes straight to the agent, bypassing STT and parser stages.
    """
    text: str


@dataclass
class AgentToken:
    """A single token yielded by the agent's streaming response."""
    text: str


@dataclass
class AgentDone:
    """Agent has finished its full response."""
    full_text: str


# ---------------------------------------------------------------------------
# TTS events
# ---------------------------------------------------------------------------


@dataclass
class TTSSpeaking:
    """TTS has started speaking the response."""
    pass


@dataclass
class TTSDone:
    """TTS has finished speaking."""
    pass


@dataclass
class TTSSentenceStart:
    """A sentence chunk has begun TTS playback."""
    text: str


@dataclass
class TTSSentenceDone:
    """A sentence chunk finished playing."""
    pass


# ---------------------------------------------------------------------------
# Tool execution events
# ---------------------------------------------------------------------------


@dataclass
class ToolExecutionStart:
    """A tool call has been started by the agent."""
    tool_call_id: str
    tool_name: str
    args: str  # JSON string of arguments


@dataclass
class ToolExecutionEnd:
    """A tool call has completed."""
    tool_call_id: str
    tool_name: str
    result: str  # JSON string of result (truncated if large)


# ---------------------------------------------------------------------------
# Error events
# ---------------------------------------------------------------------------


@dataclass
class PipelineError(Exception):
    """An error occurred at a pipeline stage."""
    stage: str   # "stt" | "parser" | "agent" | "tts"
    message: str  # human-readable description
    fatal: bool   # abort the current turn?

    def __str__(self) -> str:
        return f"[{self.stage}] {self.message}"


# ---------------------------------------------------------------------------
# Event union
# ---------------------------------------------------------------------------

Event = Union[
    InputCommand,
    STTStarted,
    STTTranscribing,
    STTDone,
    ParsingStarted,
    ParsingDone,
    UserTextInput,
    AgentToken,
    AgentDone,
    TTSSpeaking,
    TTSDone,
    TTSSentenceStart,
    TTSSentenceDone,
    ToolExecutionStart,
    ToolExecutionEnd,
    PipelineError,
]
