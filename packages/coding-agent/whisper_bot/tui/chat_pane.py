"""Scrollable chat pane with inline streaming support for the TUI.

All mount operations are scheduled via ``call_later`` so they run within Textual's
event-loop message cycle, even when called from background pipeline tasks.
"""

from __future__ import annotations

import asyncio

from textual.containers import ScrollableContainer
from textual.widgets import Static


class ChatPane(ScrollableContainer):
    """Conversation history with streaming support for agent tokens.

    User messages appear as ``[bold cyan]You:[/] <text>``.
    Agent messages appear as ``[bold green]Pi:[/] <tokens accumulate inline>``.
    Status lines appear dimmed.
    """

    def __init__(self) -> None:
        super().__init__()
        self._current_agent: Static | None = None
        self._agent_buffer = ""

    # ------------------------------------------------------------------
    # Public API (can be called from any asyncio task in the event loop)
    # ------------------------------------------------------------------

    def add_user_message(self, text: str) -> None:
        """Add a complete user message."""
        self._current_agent = None
        msg = Static(f"[bold cyan]You:[/bold cyan] {text}")
        self._schedule_mount(msg)

    def start_agent_message(self) -> None:
        """Begin a new agent message block (label + inline content)."""
        if self._current_agent is not None:
            return  # already started
        self._agent_buffer = ""
        content = Static("[bold green]Pi:[/bold green] ")
        self._current_agent = content
        self._schedule_mount(content)

    def stream_token(self, text: str) -> None:
        """Append a token to the current agent message (sync — no mount needed)."""
        if self._current_agent is None:
            self.start_agent_message()
        if self._current_agent is None:
            return
        self._agent_buffer += text
        self._current_agent.update(
            f"[bold green]Pi:[/bold green] {self._agent_buffer}"
        )

    def finalize_message(self) -> None:
        """Finalize the current agent message (stop accepting tokens)."""
        self._current_agent = None
        self._agent_buffer = ""

    def add_status(self, text: str) -> None:
        """Add a dimmed status line."""
        self._current_agent = None
        msg = Static(f"[dim]{text}[/dim]", classes="status")
        self._schedule_mount(msg)

    def clear(self) -> None:
        """Remove all messages from the pane."""
        for child in list(self.children):
            child.remove()
        self._current_agent = None

    # ------------------------------------------------------------------
    # Internal helpers — schedule DOM ops in the main event loop
    # ------------------------------------------------------------------

    def _schedule_mount(self, widget: Static) -> None:
        """Schedule a widget mount in the running event loop.

        Textual's ``mount()`` is async and must run inside the app's event
        loop.  Creating a task from the pipeline's background worker is safe
        because both share the same event loop.
        """
        async def _do_mount() -> None:
            await self.mount(widget)
            await self.scroll_end(animate=False)

        asyncio.create_task(_do_mount())
