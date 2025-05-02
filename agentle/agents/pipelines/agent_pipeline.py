from collections.abc import Sequence
from typing import Any

from rsb.models.base_model import BaseModel

from agentle.agents.agent import Agent
from agentle.agents.agent_input import AgentInput
from agentle.agents.agent_run_output import AgentRunOutput


class AgenticPipeline(BaseModel):
    agents: Sequence[Agent[Any]]

    def run(self, input: AgentInput) -> AgentRunOutput[Any]:
        raise NotImplementedError("AgenticPipeline.run is not implemented")
