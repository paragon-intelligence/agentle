from collections.abc import Sequence

from rsb.models.base_model import BaseModel

from agentle.agents.agent import Agent


class AgentSquad(BaseModel):
    agents: Sequence[Agent]
