import uuid
from collections.abc import Sequence
from typing import Any

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.messages.message import Message
from agentle.agents.models.artifact import Artifact
from agentle.agents.tasks.task_state import TaskState


class Task(BaseModel):
    """
    The central unit of work
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

    artifacts: Sequence[Artifact[Any]] | None = Field(default=None)
    """
    collection of artifacts created by the agent
    """

    metadata: dict[str, Any] | None = Field(default=None)
    """
    extension metadata
    """
