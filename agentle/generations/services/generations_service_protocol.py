from __future__ import annotations

import abc
from collections.abc import Sequence

from agentle.generations.models.generation.generation import Generation
from agentle.generations.models.messages.message import Message
from agentle.generations.providers import GenerationProvider
from rsb.coroutines.run_sync import run_sync
from rsb.decorators.services import abstractservice

type WithoutStructuredOutput = None


@abstractservice
class GenerationsServiceProtocol(abc.ABC):
    generation_provider: GenerationProvider

    def __init__(
        self,
        generation_strategy: GenerationProvider,
    ) -> None:
        self.generation_provider = generation_strategy

    def generate[T = WithoutStructuredOutput](
        self,
        model: str,
        messages: Sequence[Message],
        response_schema: type[T] | None = None,
    ) -> Generation[T]:
        return run_sync(
            self.generate_async,
            timeout=None,
            model=model,
            messages=messages,
            response_schema=response_schema,
        )

    @abc.abstractmethod
    async def generate_async[T = WithoutStructuredOutput](
        self,
        model: str,
        messages: Sequence[Message],
        response_schema: type[T] | None = None,
    ) -> Generation[T]: ...
