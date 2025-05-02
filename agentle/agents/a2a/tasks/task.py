"""
A2A Task Model

This module defines the Task class, which represents a unit of work in the A2A protocol.
Tasks encapsulate conversations between users and agents, track status, and manage artifacts.
"""

import uuid
from collections.abc import Sequence
from typing import Any

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.a2a.messages.message import Message
from agentle.agents.a2a.models.artifact import Artifact
from agentle.agents.a2a.tasks.task_state import TaskState


class Task(BaseModel):
    """
    Represents a unit of work in the A2A protocol.

    Tasks are the central concept in the A2A protocol, encapsulating conversations
    between users and agents. Each task has a unique identifier, belongs to a session,
    has a status, and contains a history of messages exchanged.

    Attributes:
        id: Unique identifier for the task
        sessionId: Identifier for the session holding the task
        status: Current status of the task (e.g., submitted, working, completed)
        history: Optional sequence of messages exchanged in the task
        artifacts: Optional artifacts created by the agent during the task
        metadata: Optional additional metadata associated with the task

    Example:
        ```python
        from agentle.agents.a2a.tasks.task import Task
        from agentle.agents.a2a.tasks.task_state import TaskState
        from agentle.agents.a2a.messages.message import Message
        from agentle.agents.a2a.message_parts.text_part import TextPart

        # Create messages for the task history
        user_message = Message(
            role="user",
            parts=[TextPart(text="Can you summarize this article?")]
        )
        agent_message = Message(
            role="agent",
            parts=[TextPart(text="Here's a summary of the article...")]
        )

        # Create a task
        task = Task(
            sessionId="session-123",
            status=TaskState.COMPLETED,
            history=[user_message, agent_message]
        )

        # Access task information
        print(f"Task ID: {task.id}")
        print(f"Task Status: {task.status}")
        print(f"Number of messages: {len(task.history) if task.history else 0}")
        ```
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    """
    Unique identifier for the task
    """

    sessionId: str
    """
    client-generated id for the session holding the task.
    """

    status: TaskState
    """
    current status of the task
    """

    history: Sequence[Message] | None = Field(default=None)
    """
    history of messages exchanged between the task and the client
    """

    artifacts: Sequence[Artifact] | None = Field(default=None)
    """
    collection of artifacts created by the agent
    """

    metadata: dict[str, Any] | None = Field(default=None)
    """
    extension metadata
    """
