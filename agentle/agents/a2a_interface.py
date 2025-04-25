from rsb.models.base_model import BaseModel

from agentle.agents.agent import Agent
from agentle.agents.resources.task_resource import TaskResource
from agentle.agents.tasks.managment.task_manager import TaskManager


class A2AInterface(BaseModel):
    agent: Agent
    task_manager: TaskManager

    @property
    def tasks(self) -> TaskResource:
        return TaskResource(agent=self.agent, manager=self.task_manager)
