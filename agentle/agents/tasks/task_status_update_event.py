import uuid
from typing import Any

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.tasks.task_status import TaskStatus


class TaskStatusUpdateEvent(BaseModel):
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
