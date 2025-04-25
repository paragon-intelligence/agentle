import random
from collections.abc import Sequence
from typing import override

from rsb.contracts.maybe_protocol import MaybeProtocol

from agentle.generations.models.generation.generation import Generation
from agentle.generations.models.generation.generation_config import GenerationConfig
from agentle.generations.models.messages.message import Message
from agentle.generations.providers.base.generation_provider import (
    GenerationProvider,
)
from agentle.generations.tools.tool import Tool
from agentle.generations.tracing.contracts.stateful_observability_client import (
    StatefulObservabilityClient,
)

type WithoutStructuredOutput = None


class FailoverGenerationProvider(GenerationProvider):
    generation_providers: Sequence[GenerationProvider]
    tracing_client: MaybeProtocol[StatefulObservabilityClient]
    shuffle: bool

    def __init__(
        self,
        tracing_client: StatefulObservabilityClient | None,
        generation_providers: Sequence[GenerationProvider],
        shuffle: bool = False,
    ) -> None:
        super().__init__(tracing_client=tracing_client)
        self.generation_providers = generation_providers
        self.shuffle = shuffle

    @property
    @override
    def organization(self) -> str:
        return "mixed"

    async def create_generation_async[T = WithoutStructuredOutput](
        self,
        *,
        model: str,
        messages: Sequence[Message],
        response_schema: type[T] | None = None,
        generation_config: GenerationConfig | None = None,
        tools: Sequence[Tool] | None = None,
    ) -> Generation[T]:
        exceptions: list[Exception] = []

        providers = list(self.generation_providers)
        if self.shuffle:
            random.shuffle(providers)

        for provider in providers:
            try:
                return await provider.create_generation_async(
                    model=model,
                    messages=messages,
                    response_schema=response_schema,
                    generation_config=generation_config,
                    tools=tools,
                )
            except Exception as e:
                exceptions.append(e)
                continue

        if not exceptions:
            raise RuntimeError("Exception is None and the for loop went out.")

        raise exceptions[0]
