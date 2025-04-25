import uuid

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.tasks.task import Task


class TaskGetResult(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    result: Task
    error: str | None = Field(default=None)
