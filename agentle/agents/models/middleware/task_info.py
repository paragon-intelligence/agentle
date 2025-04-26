from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class TaskInfo(BaseModel):
    completed: bool = Field(
        description="Whether the task is completed or not. "
        + "A task is completed if the agent has provided a response that "
        + "satisfies the task's objective."
    )
