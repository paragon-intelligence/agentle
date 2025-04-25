from rsb.models.base_model import BaseModel

from agentle.agents.context import Context
from agentle.generations.models.generation.generation import Generation


class AgentOutput[T](BaseModel):
    generation: Generation[T]
    final_context: Context

    @property
    def parsed(self) -> T:
        return self.generation.parsed
