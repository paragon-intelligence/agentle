"""
Failover mechanism for resilient AI generation across multiple providers.

This module implements a failover system for AI generation that provides fault tolerance
by trying multiple underlying providers in sequence until one succeeds. This enables
applications to maintain availability even when a specific AI provider experiences
an outage, rate limiting, or other errors.

The FailoverGenerationProvider maintains a sequence of generation providers and attempts
to use each one in order (or in random order if shuffling is enabled) until a successful
generation is produced. If all providers fail, the exception from the first provider
is raised to maintain consistent error handling.

This implementation is particularly valuable for mission-critical applications that
require high availability and cannot tolerate downtime from any single AI provider.
"""

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
    """
    Provider implementation that fails over between multiple generation providers.

    This class implements a fault-tolerant generation provider that attempts to use
    multiple underlying providers in sequence until one succeeds. If a provider raises
    an exception, the failover system catches it and tries the next provider.

    The order of providers can be either maintained as specified or randomly shuffled
    for each request if load balancing across providers is desired.

    Attributes:
        generation_providers: Sequence of underlying generation providers to use.
        tracing_client: Optional client for observability and tracing of generation
            requests and responses.
        shuffle: Whether to randomly shuffle the order of providers for each request.
    """

    generation_providers: Sequence[GenerationProvider]
    tracing_client: MaybeProtocol[StatefulObservabilityClient]
    shuffle: bool

    def __init__(
        self,
        *,
        generation_providers: Sequence[GenerationProvider],
        tracing_client: StatefulObservabilityClient | None = None,
        shuffle: bool = False,
    ) -> None:
        """
        Initialize the Failover Generation Provider.

        Args:
            tracing_client: Optional client for observability and tracing of generation
                requests and responses.
            generation_providers: Sequence of underlying generation providers to try in order.
            shuffle: Whether to randomly shuffle the order of providers for each request.
                Defaults to False (maintain the specified order).
        """
        super().__init__(tracing_client=tracing_client)
        self.generation_providers = generation_providers
        self.shuffle = shuffle

    @property
    @override
    def organization(self) -> str:
        """
        Get the provider organization identifier.

        Since this provider may use multiple underlying providers from different
        organizations, it returns a generic "mixed" identifier.

        Returns:
            str: The organization identifier, which is "mixed" for this provider.
        """
        return "mixed"

    @property
    @override
    def default_model(self) -> str:
        """
        Get the default model for the generation provider.

        Returns:
            str: The default model for the generation provider.
        """
        return self.generation_providers[0].default_model

    @override
    def price_per_million_tokens_input(
        self, model: str, estimate_tokens: int | None = None
    ) -> float:
        return 0.0

    @override
    def price_per_million_tokens_output(
        self, model: str, estimate_tokens: int | None = None
    ) -> float:
        return 0.0

    @override
    async def create_generation_async[T = WithoutStructuredOutput](
        self,
        *,
        model: str | None = None,
        messages: Sequence[Message],
        response_schema: type[T] | None = None,
        generation_config: GenerationConfig | None = None,
        tools: Sequence[Tool] | None = None,
    ) -> Generation[T]:
        """
        Create a generation with failover across multiple providers.

        This method attempts to create a generation using each provider in sequence
        until one succeeds. If a provider raises an exception, it catches the exception
        and tries the next provider. If all providers fail, it raises the first exception.

        Args:
            model: The model identifier to use for generation.
            messages: A sequence of Message objects to send to the model.
            response_schema: Optional Pydantic model for structured output parsing.
            generation_config: Optional configuration for the generation request.
            tools: Optional sequence of Tool objects for function calling.

        Returns:
            Generation[T]: An Agentle Generation object from the first successful provider.

        Raises:
            Exception: The exception from the first provider if all providers fail.
        """
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
