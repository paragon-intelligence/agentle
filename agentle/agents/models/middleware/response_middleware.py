from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.models.middleware.task_info import TaskInfo
from agentle.agents.tasks.task_state import TaskState

type StringOutput = str


class ResponseMiddleware[T_Schema = StringOutput](BaseModel):
    task: str = Field(description="The identified task that the agent must complete. ")
    response: T_Schema
    task_info: TaskInfo = Field(
        description="Information about the task. "
        + "A task is the unit of work that the agent is performing, like answering a question or "
        + "completing a task."
    )

    @property
    def task_completed(self) -> bool:
        return self.task_info.state == TaskState.COMPLETED
