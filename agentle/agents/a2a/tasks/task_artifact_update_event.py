"""
A2A Task Artifact Update Event Model

This module defines the TaskArtifactUpdateEvent class, which represents an event
sent when a task generates or updates an artifact in the A2A protocol. It enables
clients to be notified of artifact creation and updates asynchronously.
"""

from typing import Any

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.a2a.models.artifact import Artifact


class TaskArtifactUpdateEvent(BaseModel):
    """
    Represents an event sent when a task generates or updates an artifact.

    This class encapsulates information about an artifact update, allowing clients
    to be notified asynchronously when a task creates or modifies artifacts. It includes
    the task ID, the artifact data, and additional metadata.

    Attributes:
        id: The identifier of the task
        artifact: The artifact created or updated by the agent
        metadata: Additional metadata about the artifact update

    Example:
        ```python
        from agentle.agents.a2a.tasks.task_artifact_update_event import TaskArtifactUpdateEvent
        from agentle.agents.a2a.models.artifact import Artifact
        from agentle.generations.models.message_parts.text import TextPart

        # Create an artifact
        text_part = TextPart(text="def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a")
        artifact = Artifact(
            name="fibonacci.py",
            description="Python function to calculate Fibonacci numbers",
            parts=[text_part],
            index=0
        )

        # Create an artifact update event
        event = TaskArtifactUpdateEvent(
            id="task-123",
            artifact=artifact,
            metadata={"generation_time_ms": 1200}
        )

        # Access event information
        print(f"Task ID: {event.id}")
        print(f"Artifact Name: {event.artifact.name}")
        print(f"Artifact Description: {event.artifact.description}")
        print(f"Metadata: {event.metadata}")
        ```

    Note:
        Artifacts can be generated in chunks for large outputs. In such cases,
        multiple TaskArtifactUpdateEvent instances may be sent for the same artifact,
        with the `append` flag set on the artifact to indicate that the content
        should be appended to an existing artifact.
    """

    id: str = Field(...)
    """
    Task id
    """

    artifact: Artifact
    """
    artifact created by the agent
    """

    metadata: dict[str, Any]
    """
    extension metadata
    """
