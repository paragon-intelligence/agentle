"""
A2A Task Status Update Event Model

This module defines the TaskStatusUpdateEvent class, which represents an event
sent when a task's status changes in the A2A protocol. It enables clients to be
notified of task status changes asynchronously.
"""

import uuid
from typing import Any

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.a2a.tasks.task_status import TaskStatus


class TaskStatusUpdateEvent(BaseModel):
    """
    Represents an event sent when a task's status changes.

    This class encapsulates information about a task status change, allowing clients
    to be notified asynchronously when a task transitions between states. It includes
    the task ID, the new status, a flag indicating if this is the final status update,
    and additional metadata.

    Attributes:
        id: The identifier of the task
        status: The new status of the task
        final: Flag indicating if this is the final status update
        metadata: Additional metadata about the status update

    Example:
        ```python
        from agentle.agents.a2a.tasks.task_status_update_event import TaskStatusUpdateEvent
        from agentle.agents.a2a.tasks.task_status import TaskStatus
        from agentle.agents.a2a.tasks.task_state import TaskState
        from agentle.agents.a2a.messages.message import Message
        from agentle.agents.a2a.message_parts.text_part import TextPart
        from datetime import datetime, timezone

        # Create a message for the status
        message = Message(
            role="agent",
            parts=[TextPart(text="Task completed successfully.")]
        )

        # Create a task status
        status = TaskStatus(
            state=TaskState.COMPLETED,
            message=message,
            timestamp=datetime.now(timezone.utc)
        )

        # Create a status update event
        event = TaskStatusUpdateEvent(
            id="task-123",
            status=status,
            final=True,
            metadata={"completion_time_ms": 1500}
        )

        # Access event information
        print(f"Task ID: {event.id}")
        print(f"Task State: {event.status.state}")
        print(f"Is Final Update: {event.final}")
        print(f"Metadata: {event.metadata}")
        ```

    Note:
        When the `final` flag is set to True, clients should consider the task
        lifecycle complete and not expect any further status updates for that task.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    """
    Task id
    """

    status: TaskStatus
    """
    Status of the task
    """

    final: bool
    """
    indicates the end of the event stream
    """

    metadata: dict[str, Any]
    """
    Additional metadata about the status update
    """
