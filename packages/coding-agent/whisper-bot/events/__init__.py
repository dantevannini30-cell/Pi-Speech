"""Event types and bus for the whisper-bot pipeline."""

from whisper_bot.events.bus import EventBus
from whisper_bot.events.types import (
    AgentDone,
    AgentToken,
    Event,
    InputCommand,
    ParsingDone,
    ParsingStarted,
    PipelineError,
    STTDone,
    STTStarted,
    STTTranscribing,
    TTSDone,
    TTSSpeaking,
    TTSSentenceStart,
    TTSSentenceDone,
    ToolExecutionEnd,
    ToolExecutionStart,
    UserTextInput,
)

__all__ = [
    "EventBus",
    "AgentDone",
    "AgentToken",
    "Event",
    "InputCommand",
    "ParsingDone",
    "ParsingStarted",
    "PipelineError",
    "STTDone",
    "STTStarted",
    "STTTranscribing",
    "TTSDone",
    "TTSSpeaking",
    "TTSSentenceStart",
    "TTSSentenceDone",
    "ToolExecutionEnd",
    "ToolExecutionStart",
    "UserTextInput",
]
