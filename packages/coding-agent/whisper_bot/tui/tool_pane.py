"""Collapsible tool pane showing Pi's tool calls as they execute."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.widgets import Header, Static
from textual.widget import Widget


class ToolPane(Widget):
    """Side pane showing tool execution calls and results."""

    DEFAULT_CSS = """
    ToolPane {
        background: $surface;
        border-left: solid $primary;
        padding: 0 1;
        overflow-y: auto;
    }
    ToolPane.hidden {
        display: none;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header("Tools", classes="tool-header")

    def add_tool_call(self, tool_name: str, args: str) -> None:
        """Add a tool call entry to the pane."""
        display_args = (args[:100] + "...") if len(args) > 100 else args
        call = Static(
            f"[bold]{tool_name}[/bold] {display_args}",
            classes="tool-call",
        )
        self.mount(call)
        self.scroll_end(animate=False)

    def finish_tool_call(self, tool_call_id: str, result: str) -> None:
        """Mark the most recent tool call as completed."""
        # Find the latest unmatched tool-call and mark it done
        for child in reversed(list(self.children)):
            if (
                isinstance(child, Static)
                and "tool-call" in (child.classes or "")
            ):
                current = child.renderable or ""
                if not current.endswith(" [dim]done[/dim]"):
                    child.update(f"{current} [dim]done[/dim]")
                break

    def toggle_visible(self) -> None:
        """Toggle the hidden class on/off."""
        self.set_class(not self.has_class("hidden"), "hidden")
