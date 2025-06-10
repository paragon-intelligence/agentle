from rsb.models.base_model import BaseModel
from rsb.models.private_attr import PrivateAttr

from agentle.agents.agent import Agent
from agentle.generations.providers.base.generation_provider import GenerationProvider


class Webson[T](BaseModel):
    generation_provider: GenerationProvider
    cast_to: T
    _agent: Agent[T] = PrivateAttr()

    def __post_init__(self) -> None:
        self._agent = Agent(generation_provider=self.generation_provider)

    async def cast_async(self, page: str) -> T: ...
