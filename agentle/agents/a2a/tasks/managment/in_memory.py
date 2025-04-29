from typing import ClassVar

from rsb.models.field import Field

from agentle.agents.a2a.tasks.managment.task_manager import TaskManager
from agentle.agents.a2a.tasks.task import Task


class InMemoryTaskManager(TaskManager):
    tasks: ClassVar[list[Task]] = Field(default_factory=list)
