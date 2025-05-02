"""
A2A Task Status Model

This module defines the TaskStatus class, which represents the status of a task
in the A2A protocol. It encapsulates the current state, message, and timestamp
information for a task, providing detailed status tracking.
"""

from datetime import datetime

from rsb.models.base_model import BaseModel

from agentle.agents.a2a.messages.message import Message
from agentle.agents.a2a.tasks.task_state import TaskState


class TaskStatus(BaseModel):
    """
    Represents the detailed status of a task in the A2A protocol.

    This class encapsulates the current state, message, and timestamp information
    for a task, providing detailed status tracking. It is used for monitoring task
    progress and communicating status updates to clients.

    Attributes:
        state: The current state of the task (e.g., submitted, working, completed)
        message: The message associated with the current status
        timestamp: The timestamp when the status was updated

    Example:
        ```python
        from agentle.agents.a2a.tasks.task_status import TaskStatus
        from agentle.agents.a2a.tasks.task_state import TaskState
        from agentle.agents.a2a.messages.message import Message
        from agentle.agents.a2a.message_parts.text_part import TextPart
        from datetime import datetime, timezone

        # Create a message for the status
        message = Message(
            role="agent",
            parts=[TextPart(text="Processing your request...")]
        )

        # Create a task status
        status = TaskStatus(
            state=TaskState.WORKING,
            message=message,
            timestamp=datetime.now(timezone.utc)
        )

        # Access status information
        print(f"Task State: {status.state}")
        print(f"Status Time: {status.timestamp.isoformat()}")
        if status.message.parts and len(status.message.parts) > 0:
            for part in status.message.parts:
                if part.type == "text":
                    print(f"Status Message: {part.text}")
        ```

    Note:
        The timestamp should be in UTC to ensure consistent time representation
        across different systems and timezones.
    """

    state: TaskState
    """
    additional status updates for client
    """

    message: Message
    """
    additional status updates for client
    """

    timestamp: datetime
    """
    ISO datetime value
    """
