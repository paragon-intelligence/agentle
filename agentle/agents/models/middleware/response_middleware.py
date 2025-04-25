from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.models.middleware.task_info import TaskInfo

type StringOutput = str


class ResponseMiddleware[T_Schema = StringOutput](BaseModel):
    response: T_Schema
    task_info: TaskInfo = Field(description="Information about the task")
