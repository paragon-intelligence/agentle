"""
A2A Tasks Package

This package provides classes for managing tasks in the A2A protocol. Tasks are the central
unit of work in the A2A protocol, representing conversations between users and agents and
tracking their state and progress.

The package includes:
- Task models and state tracking
- Task management functionality
- Task parameter models for sending and querying
- Task result models for responses
- Event models for task updates

Example:
    ```python
    from agentle.agents.a2a.tasks.task import Task
    from agentle.agents.a2a.tasks.task_state import TaskState
    from agentle.agents.a2a.messages.message import Message
    from agentle.agents.a2a.message_parts.text_part import TextPart

    # Create a message
    message = Message(
        role="user",
        parts=[TextPart(text="Hello, can you help me?")]
    )

    # Create a task
    task = Task(
        sessionId="session-123",
        status=TaskState.SUBMITTED,
        history=[message]
    )

    # Check task status
    print(f"Task ID: {task.id}")
    print(f"Task Status: {task.status}")
    ```
"""

from agentle.agents.a2a.tasks.task import Task
from agentle.agents.a2a.tasks.task_state import TaskState
from agentle.agents.a2a.tasks.task_status import TaskStatus
from agentle.agents.a2a.tasks.task_send_params import TaskSendParams
from agentle.agents.a2a.tasks.task_query_params import TaskQueryParams
from agentle.agents.a2a.tasks.task_get_result import TaskGetResult
from agentle.agents.a2a.tasks.task_artifact_update_event import TaskArtifactUpdateEvent
from agentle.agents.a2a.tasks.task_status_update_event import TaskStatusUpdateEvent
from agentle.agents.a2a.tasks.send_task_response import SendTaskResponse

__all__ = [
    "Task",
    "TaskState",
    "TaskStatus",
    "TaskSendParams",
    "TaskQueryParams",
    "TaskGetResult",
    "TaskArtifactUpdateEvent",
    "TaskStatusUpdateEvent",
    "SendTaskResponse",
]
