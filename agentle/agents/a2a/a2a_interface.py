from typing import Any
from rsb.models.base_model import BaseModel

from agentle.agents.a2a.resources.task_resource import TaskResource
from agentle.agents.a2a.tasks.managment.task_manager import TaskManager
from agentle.agents.agent import Agent
from agentle.agents.agent_pipeline import AgentPipeline
from agentle.agents.agent_team import AgentTeam


class A2AInterface[T_Schema = Any](BaseModel):
    agent: Agent[T_Schema] | AgentTeam | AgentPipeline
    task_manager: TaskManager

    @property
    def tasks(self) -> TaskResource[T_Schema]:
        return TaskResource(agent=self.agent, manager=self.task_manager)
