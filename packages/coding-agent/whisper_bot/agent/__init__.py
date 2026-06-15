"""Agent provider interface.

Agents receive cleaned user input and produce streaming text responses.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class AgentProvider(ABC):
    """Interface for conversational agents."""

    @abstractmethod
    async def process(self, text: str) -> AsyncIterator[str]:
        """Process *text* and yield response tokens as they arrive."""
        ...
        # A dummy yield so the generator machinery fires even if a subclass
        # never yields (avoids a runtime warning).  Real subclasses must
        # override this method fully.
        if False:  # pragma: no cover
            yield ""

    async def shutdown(self) -> None:
        """Clean up any resources held by the provider.

        Called when the pipeline is shutting down.  Default is a no-op.
        Subclasses that hold subprocesses, network connections, or other
        resources should override this method.
        """
        pass
