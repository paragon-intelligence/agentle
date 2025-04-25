from typing import ClassVar

from rsb.models.field import Field

from agentle.agents.tasks.managment.task_manager import TaskManager
from agentle.agents.tasks.task import Task


class InMemoryTaskManager(TaskManager):
    tasks: ClassVar[list[Task]] = Field(default_factory=list)
