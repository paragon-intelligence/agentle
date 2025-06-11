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

from __future__ import annotations

import random
from collections.abc import MutableSequence, Sequence
from typing import override

from rsb.contracts.maybe_protocol import MaybeProtocol
from rsb.coroutines.fire_and_forget import fire_and_forget

from agentle.generations.models.generation.generation import Generation
from agentle.generations.models.generation.generation_config import GenerationConfig
from agentle.generations.models.generation.generation_config_dict import (
    GenerationConfigDict,
)
from agentle.generations.models.messages.message import Message
from agentle.generations.providers.base.generation_provider import (
    GenerationProvider,
)
from agentle.generations.providers.types.model_kind import ModelKind
from agentle.generations.tools.tool import Tool
from agentle.generations.tracing.contracts.stateful_observability_client import (
    StatefulObservabilityClient,
)
from agentle.resilience.circuit_breaker.circuit_breaker_protocol import (
    CircuitBreakerProtocol,
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

    Optionally supports circuit breaker pattern to avoid repeatedly trying providers
    that have recently failed, improving response times in production scenarios.

    Attributes:
        generation_providers: Sequence of underlying generation providers to use.
        tracing_client: Optional client for observability and tracing of generation
            requests and responses.
        shuffle: Whether to randomly shuffle the order of providers for each request.
        circuit_breaker: Optional circuit breaker to track provider failures and
            temporarily skip failing providers.
    """

    generation_providers: Sequence[GenerationProvider]
    tracing_client: MaybeProtocol[StatefulObservabilityClient]
    shuffle: bool
    circuit_breaker: CircuitBreakerProtocol | None

    def __init__(
        self,
        *,
        generation_providers: Sequence[
            GenerationProvider | Sequence[GenerationProvider]
        ],
        tracing_client: StatefulObservabilityClient | None = None,
        shuffle: bool = False,
        circuit_breaker: CircuitBreakerProtocol | None = None,
    ) -> None:
        """
        Initialize the Failover Generation Provider.

        Args:
            tracing_client: Optional client for observability and tracing of generation
                requests and responses.
            generation_providers: Sequence of underlying generation providers or sequences
                of providers to try in order. Nested sequences will be flattened.
            shuffle: Whether to randomly shuffle the order of providers for each request.
                Defaults to False (maintain the specified order).
            circuit_breaker: Optional circuit breaker to track provider failures and
                temporarily skip failing providers. If None, circuit breaker logic is disabled.
        """
        super().__init__(tracing_client=tracing_client)

        # Flatten nested sequences of providers
        flattened_providers: MutableSequence[GenerationProvider] = []
        for item in generation_providers:
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                # If it's a sequence (but not string/bytes), extend with its contents
                flattened_providers.extend(item)
            else:
                # If it's a single provider, append it
                flattened_providers.append(item)

        self.generation_providers = flattened_providers
        self.shuffle = shuffle
        self.circuit_breaker = circuit_breaker

    def _get_provider_circuit_id(self, provider: GenerationProvider) -> str:
        """
        Generate a unique identifier for a provider to use as circuit breaker key.

        Uses a combination of provider class name, organization, and instance id
        to create a reasonably unique identifier for circuit tracking.
        """
        # Use provider instance id as base
        circuit_id = f"{provider.__class__.__name__}_{id(provider)}"

        # Add organization if available for better grouping
        if hasattr(provider, "organization"):
            circuit_id = f"{provider.organization}_{circuit_id}"

        return circuit_id

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
        model: str | ModelKind | None = None,
        messages: Sequence[Message],
        response_schema: type[T] | None = None,
        generation_config: GenerationConfig | GenerationConfigDict | None = None,
        tools: Sequence[Tool] | None = None,
    ) -> Generation[T]:
        """
        Create a generation with failover across multiple providers.

        This method attempts to create a generation using each provider in sequence
        until one succeeds. If a provider raises an exception, it catches the exception
        and tries the next provider. If all providers fail, it raises the first exception.

        When a circuit breaker is configured, providers with open circuits are skipped
        to avoid unnecessary delays from repeatedly trying failing providers.

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
        exceptions: MutableSequence[tuple[GenerationProvider, Exception]] = []

        # Get list of providers and optionally shuffle
        providers = list(self.generation_providers)
        if self.shuffle:
            random.shuffle(providers)

        # Track which providers were skipped due to open circuits
        skipped_providers: MutableSequence[GenerationProvider] = []

        for provider in providers:
            # Check circuit breaker if configured
            if self.circuit_breaker is not None:
                circuit_id = self._get_provider_circuit_id(provider)

                # Check if circuit is open
                try:
                    is_open = await self.circuit_breaker.is_open(circuit_id)
                    if is_open:
                        # Skip this provider as its circuit is open
                        skipped_providers.append(provider)
                        continue
                except Exception:
                    # If circuit breaker check fails, proceed with the provider
                    # This ensures we don't break functionality if circuit breaker has issues
                    pass

            try:
                # Attempt generation with this provider
                result = await provider.create_generation_async(
                    model=model,
                    messages=messages,
                    response_schema=response_schema,
                    generation_config=generation_config,
                    tools=tools,
                )

                # Success - record it if circuit breaker is configured
                if self.circuit_breaker is not None:
                    circuit_id = self._get_provider_circuit_id(provider)
                    try:
                        fire_and_forget(self.circuit_breaker.record_success, circuit_id)
                    except Exception:
                        # Don't fail the successful generation if circuit breaker has issues
                        pass

                return result

            except Exception as e:
                # Record the failure
                exceptions.append((provider, e))

                # Record failure to circuit breaker if configured
                if self.circuit_breaker is not None:
                    circuit_id = self._get_provider_circuit_id(provider)
                    try:
                        fire_and_forget(self.circuit_breaker.record_failure, circuit_id)
                    except Exception:
                        # Don't fail if circuit breaker has issues
                        pass

                continue

        # If we skipped some providers due to open circuits and all others failed,
        # we might want to retry the skipped ones as a last resort
        if skipped_providers and exceptions:
            for provider in skipped_providers:
                try:
                    result = await provider.create_generation_async(
                        model=model,
                        messages=messages,
                        response_schema=response_schema,
                        generation_config=generation_config,
                        tools=tools,
                    )

                    # Success - the circuit might be recovering
                    if self.circuit_breaker is not None:
                        circuit_id = self._get_provider_circuit_id(provider)
                        try:
                            fire_and_forget(
                                self.circuit_breaker.record_success, circuit_id
                            )
                        except Exception:
                            pass

                    return result

                except Exception as e:
                    exceptions.append((provider, e))
                    # Circuit is still failing, it will remain open
                    continue

        # All providers failed
        if not exceptions:
            raise RuntimeError(
                "No providers available. All providers were skipped due to open circuits."
            )

        raise exceptions[0][1]

    @override
    def map_model_kind_to_provider_model(
        self,
        model_kind: ModelKind,
    ) -> str:
        raise NotImplementedError(
            "This method should not be called on the FailoverGenerationProvider."
        )

    def without_provider_type(
        self, provider_type: type[GenerationProvider]
    ) -> FailoverGenerationProvider:
        """
        Create a new FailoverGenerationProvider without providers of the specified type.

        This method recursively removes providers of the specified type from nested
        FailoverGenerationProviders as well.

        Args:
            provider_type: The generation provider type to remove from the failover sequence.

        Returns:
            FailoverGenerationProvider: A new instance with all providers of the specified type removed.
        """
        filtered_providers: MutableSequence[GenerationProvider] = []

        for provider in self.generation_providers:
            if isinstance(provider, provider_type):
                # Skip providers of the target type
                continue
            elif isinstance(provider, FailoverGenerationProvider):
                # Recursively filter nested failover providers
                nested_filtered = provider.without_provider_type(provider_type)
                # Only add if it still has providers after filtering
                if nested_filtered.generation_providers:
                    filtered_providers.append(nested_filtered)
            else:
                # Keep other provider types
                filtered_providers.append(provider)

        return FailoverGenerationProvider(
            generation_providers=filtered_providers,
            tracing_client=self.tracing_client.unwrap()
            if self.tracing_client
            else None,
            shuffle=self.shuffle,
            circuit_breaker=self.circuit_breaker,
        )

    def __sub__(
        self,
        other: GenerationProvider
        | type[GenerationProvider]
        | Sequence[GenerationProvider | type[GenerationProvider]],
    ) -> FailoverGenerationProvider:
        """
        Remove providers or provider types from the failover sequence.

        This method supports removing:
        - A specific provider instance
        - All providers of a specific type
        - Multiple providers/types from a sequence

        Args:
            other: The provider(s) or provider type(s) to remove from the failover sequence.

        Returns:
            FailoverGenerationProvider: A new instance with the specified providers removed.
        """
        filtered_providers: MutableSequence[GenerationProvider] = []

        for provider in self.generation_providers:
            should_remove = False

            # Check if this provider should be removed
            if isinstance(other, (list, tuple)):
                # Handle sequence of items to remove
                for item in other:
                    if isinstance(item, type):
                        # Remove by type
                        if isinstance(provider, item):
                            should_remove = True
                            break
                    else:
                        # Remove by instance
                        if provider is item:
                            should_remove = True
                            break
            else:
                # Handle single item to remove
                if isinstance(other, type):
                    # Remove by type
                    if isinstance(provider, other):
                        should_remove = True
                else:
                    # Remove by instance
                    if provider is other:
                        should_remove = True

            if should_remove:
                continue

            # Handle nested FailoverGenerationProviders recursively
            if isinstance(provider, FailoverGenerationProvider):
                nested_filtered = provider.__sub__(other)
                # Only add if it still has providers after filtering
                if nested_filtered.generation_providers:
                    filtered_providers.append(nested_filtered)
            else:
                filtered_providers.append(provider)

        return FailoverGenerationProvider(
            generation_providers=filtered_providers,
            tracing_client=self.tracing_client.unwrap()
            if self.tracing_client
            else None,
            shuffle=self.shuffle,
            circuit_breaker=self.circuit_breaker,
        )
