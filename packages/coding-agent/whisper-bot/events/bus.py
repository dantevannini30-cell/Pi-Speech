"""Async event bus — single asyncio.Queue with subscribe helpers."""

from __future__ import annotations
from whisper_bot.debug import log as _debug

import asyncio
import sys
from collections.abc import AsyncIterator, Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from whisper_bot.events.types import Event


class EventBus:
    """Unified event bus backed by a single asyncio.Queue.

    One consumer (display) subscribes via ``subscribe()``.  Additional
    synchronous callbacks can be registered with ``subscribe_sync()`` for
    logging or side-effects.
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[Event] = asyncio.Queue()
        self._sync_handlers: list[Callable[[Event], None]] = []
        _debug("[DEBUG bus] EventBus created")

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    async def emit(self, event: Event) -> None:
        """Push an event onto the bus."""
        event_type = type(event).__name__
        event_repr = repr(event)
        # Truncate long reprs for readability
        if len(event_repr) > 300:
            event_repr = event_repr[:300] + "..."
        _debug(f"[DEBUG bus] emit({event_type}): {event_repr}")
        self._queue.put_nowait(event)
        for handler in self._sync_handlers:
            try:
                handler(event)
            except Exception as e:
                _debug(f"[DEBUG bus] Sync handler error: {e}")

    # ------------------------------------------------------------------
    # Subscribe — async consumer
    # ------------------------------------------------------------------

    async def subscribe(self) -> AsyncIterator[Event]:
        """Async generator that yields events as they arrive.

        Never ends — break out of the loop to stop consuming.
        """
        _debug("[DEBUG bus] subscribe() generator started, waiting for events...")
        while True:
            event = await self._queue.get()
            event_type = type(event).__name__
            _debug(f"[DEBUG bus] subscribe() yielding event: {event_type}")
            yield event

    # ------------------------------------------------------------------
    # Subscribe — synchronous callback
    # ------------------------------------------------------------------

    def subscribe_sync(self, handler: Callable[[Event], None]) -> None:
        """Register a synchronous callback invoked for every event."""
        self._sync_handlers.append(handler)
        _debug(f"[DEBUG bus] Sync handler registered: {handler.__name__ if hasattr(handler, '__name__') else handler}")