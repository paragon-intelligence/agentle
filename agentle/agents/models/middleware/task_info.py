from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class TaskInfo(BaseModel):
    completed: bool = Field(
        description="Whether the task is completed or not. If true, the agent will stop executing."
    )
