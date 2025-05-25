from collections.abc import MutableMapping
from datetime import datetime
from typing import Any

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.context_state import ContextState


class ExecutionState(BaseModel):
    """
    Tracks the execution state of an agent context.

    This class maintains information about the current state of execution,
    including iteration counts, timing, and resumption capabilities.
    """

    state: ContextState = Field(default="initialized")
    """Current state of the context execution."""

    current_iteration: int = Field(default=0)
    """Current iteration number in the agent execution loop."""

    max_iterations: int = Field(default=10)
    """Maximum number of iterations allowed."""

    total_tool_calls: int = Field(default=0)
    """Total number of tool calls made across all iterations."""

    started_at: datetime | None = Field(default=None)
    """When the execution started."""

    last_updated_at: datetime = Field(default_factory=datetime.now)
    """When the context was last updated."""

    paused_at: datetime | None = Field(default=None)
    """When the execution was paused (for resumption)."""

    completed_at: datetime | None = Field(default=None)
    """When the execution completed."""

    total_duration_ms: float | None = Field(default=None)
    """Total execution time in milliseconds."""

    pause_reason: str | None = Field(default=None)
    """Reason why execution was paused."""

    error_message: str | None = Field(default=None)
    """Error message if execution failed."""

    resumable: bool = Field(default=True)
    """Whether this context can be resumed after pausing."""

    checkpoint_data: MutableMapping[str, Any] = Field(default_factory=dict)
    """Additional data needed for resumption."""
