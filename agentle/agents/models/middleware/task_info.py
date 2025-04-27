from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.tasks.task_state import TaskState


class TaskInfo(BaseModel):
    state: TaskState = Field(description="The current state of the task. ")
