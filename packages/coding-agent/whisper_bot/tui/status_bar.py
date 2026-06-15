"""Status bar widget for the TUI — shows recording/TTS state."""

from __future__ import annotations

from textual.reactive import var
from textual.widgets import Static


class WhisperStatusBar(Static):
    """Bottom status bar showing recording and TTS state."""

    recording = var(False)
    tts_enabled = var(True)

    def watch_recording(self, val: bool) -> None:
        """React to recording state changes."""
        self._update()

    def watch_tts_enabled(self, val: bool) -> None:
        """React to TTS enabled state changes."""
        self._update()

    def _update(self) -> None:
        """Re-render the status bar content."""
        rec = "[bold red]REC ●[/bold red]" if self.recording else "       "
        tts = "[green]on[/green]" if self.tts_enabled else "[red]off[/red]"
        self.update(
            f" {rec}  TTS: {tts}   "
            f"Space=record  Ctrl+T=TTS  Ctrl+E=tools  Ctrl+L=clear  Ctrl+C=quit"
        )
