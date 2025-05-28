"""
Abstract base class defining the contract for AI generation providers in Agentle.

This module defines the GenerationProvider abstract base class, which serves as the
foundation for all AI provider implementations in the Agentle framework. It establishes
a common interface that all providers must implement, ensuring consistency across
different AI services.

The GenerationProvider abstract class defines methods for generating AI completions
from both prompts and message sequences, supporting both synchronous and asynchronous
execution patterns. It also includes support for structured output parsing through
generic type parameters and tool/function calling capabilities.

Provider implementations (such as for OpenAI, Google, Anthropic, etc.) inherit from
this base class and implement the abstract methods according to each provider's specific
API requirements, while maintaining the common interface for framework consumers.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Never, cast

from rsb.containers.maybe import Maybe
from rsb.contracts.maybe_protocol import MaybeProtocol
from rsb.coroutines.run_sync import run_sync

from agentle.generations.models.generation.generation import Generation
from agentle.generations.models.generation.generation_config import GenerationConfig
from agentle.generations.models.generation.generation_config_dict import (
    GenerationConfigDict,
)
from agentle.generations.models.message_parts.file import FilePart
from agentle.generations.models.message_parts.part import Part
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.message_parts.tool_execution_suggestion import (
    ToolExecutionSuggestion,
)
from agentle.generations.models.messages.assistant_message import AssistantMessage
from agentle.generations.models.messages.developer_message import DeveloperMessage
from agentle.generations.models.messages.message import Message
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.providers.types.model_kind import ModelKind
from agentle.generations.tools.tool import Tool
from agentle.generations.tracing.contracts.stateful_observability_client import (
    StatefulObservabilityClient,
)
from agentle.prompts.models.prompt import Prompt

type WithoutStructuredOutput = None

if TYPE_CHECKING:
    from agentle.generations.providers.failover.failover_generation_provider import (
        FailoverGenerationProvider,
    )


class GenerationProvider(abc.ABC):
    """
    Abstract base class for AI generation service providers.

    This class defines the interface that all AI provider implementations must adhere to
    in the Agentle framework. It provides methods for generating AI completions from both
    simple prompts and structured message sequences, supporting synchronous and asynchronous
    patterns.

    The class is generic over type T, which represents the optional structured data format
    that can be extracted from model responses when a response_schema is provided.

    Attributes:
        tracing_client: An optional client for observability and tracing of generation requests.
        default_model: An optional default model to use for generation.
    """

    tracing_client: MaybeProtocol[StatefulObservabilityClient]

    def __init__(
        self,
        *,
        tracing_client: StatefulObservabilityClient | None = None,
    ) -> None:
        """
        Initialize the generation provider.

        Args:
            tracing_client: Optional client for observability and tracing of generation
                requests and responses.
            default_model: Optional default model to use for generation.
        """
        self.tracing_client = Maybe(tracing_client)

    @property
    @abc.abstractmethod
    def default_model(self) -> str:
        """
        The default model to use for generation.
        """
        ...

    @property
    @abc.abstractmethod
    def organization(self) -> str:
        """
        Get the organization identifier for this provider.

        This property should return a string that uniquely identifies the AI provider
        organization (e.g., "openai", "google", "anthropic").

        Returns:
            str: The organization identifier.
        """
        ...

    def create_generation_by_prompt[T = WithoutStructuredOutput](
        self,
        *,
        model: str | ModelKind | None = None,
        prompt: str | Prompt | Part | Sequence[Part],
        developer_prompt: str | Prompt | None = None,
        response_schema: type[T] | None = None,
        generation_config: GenerationConfig | GenerationConfigDict | None = None,
        tools: Sequence[Tool] | None = None,
    ) -> Generation[T]:
        """
        Create a generation from a prompt synchronously.

        This is a convenience method that converts prompt-based inputs into a message-based
        format and calls the asynchronous implementation synchronously.

        Args:
            model: The model identifier to use for generation.
            prompt: The user's prompt as a string, Prompt object, or sequence of Parts.
            developer_prompt: The system/developer prompt as a string or Prompt object.
            response_schema: Optional Pydantic model for structured output parsing.
            generation_config: Optional configuration for the generation request.
            tools: Optional sequence of Tool objects for function calling.

        Returns:
            Generation[T]: An Agentle Generation object containing the model's response,
                potentially with structured output if a response_schema was provided.
        """
        _generation_config = self._normalize_generation_config(generation_config)

        return run_sync(
            self.create_generation_by_prompt_async,
            timeout=_generation_config.timeout
            if _generation_config.timeout
            else _generation_config.timeout_s * 1000
            if _generation_config.timeout_s
            else _generation_config.timeout_m * 60 * 1000
            if _generation_config.timeout_m
            else None,
            model=model,
            prompt=prompt,
            developer_prompt=developer_prompt,
            response_schema=response_schema,
            generation_config=generation_config,
            tools=tools,
        )

    async def create_generation_by_prompt_async[T = WithoutStructuredOutput](
        self,
        *,
        model: str | ModelKind | None = None,
        prompt: str | Prompt | Part | Sequence[Part],
        developer_prompt: str | Prompt | None = None,
        response_schema: type[T] | None = None,
        generation_config: GenerationConfig | GenerationConfigDict | None = None,
        tools: Sequence[Tool] | None = None,
    ) -> Generation[T]:
        """
        Create a generation from a prompt asynchronously.

        This method converts various prompt formats into a structured message sequence
        and calls the create_generation_async method with the converted messages.

        Args:
            model: The model identifier to use for generation.
            prompt: The user's prompt as a string, Prompt object, or sequence of Parts.
            developer_prompt: The system/developer prompt as a string or Prompt object.
            response_schema: Optional Pydantic model for structured output parsing.
            generation_config: Optional configuration for the generation request.
            tools: Optional sequence of Tool objects for function calling.

        Returns:
            Generation[T]: An Agentle Generation object containing the model's response,
                potentially with structured output if a response_schema was provided.
        """
        user_message_parts: Sequence[Part]
        match prompt:
            case str():
                user_message_parts = cast(Sequence[Part], [TextPart(text=prompt)])
            case Prompt():
                user_message_parts = cast(
                    Sequence[Part], [TextPart(text=prompt.content)]
                )
            case TextPart() | FilePart() | Tool() | ToolExecutionSuggestion():
                user_message_parts = cast(Sequence[Part], [prompt])
            case _:
                user_message_parts = prompt

        developer_message_parts: Sequence[Part]
        match developer_prompt:
            case str():
                developer_message_parts = [TextPart(text=developer_prompt)]
            case Prompt():
                developer_message_parts = [TextPart(text=developer_prompt.content)]
            case _:
                developer_message_parts = [
                    TextPart(text="You are a helpful assistant.")
                ]

        user_message = UserMessage(parts=user_message_parts)
        developer_message = DeveloperMessage(
            parts=cast(Sequence[TextPart], developer_message_parts)
        )

        return await self.create_generation_async(
            model=model,
            messages=[developer_message, user_message],
            response_schema=response_schema,
            generation_config=generation_config,
            tools=tools,
        )

    def create_generation[T = WithoutStructuredOutput](
        self,
        *,
        model: str | ModelKind | None = None,
        messages: Sequence[Message],
        response_schema: type[T] | None = None,
        generation_config: GenerationConfig | GenerationConfigDict | None = None,
        tools: Sequence[Tool] | None = None,
    ) -> Generation[T]:
        """
        Create a generation from a message sequence synchronously.

        This is a convenience method that calls the asynchronous implementation
        synchronously using a wrapper.

        Args:
            model: The model identifier to use for generation.
            messages: A sequence of structured Message objects to send to the model.
            response_schema: Optional Pydantic model for structured output parsing.
            generation_config: Optional configuration for the generation request.
            tools: Optional sequence of Tool objects for function calling.

        Returns:
            Generation[T]: An Agentle Generation object containing the model's response,
                potentially with structured output if a response_schema was provided.
        """
        _generation_config = self._normalize_generation_config(generation_config)

        return run_sync(
            self.create_generation_async,
            timeout=_generation_config.timeout
            if _generation_config.timeout
            else _generation_config.timeout_s * 1000
            if _generation_config.timeout_s
            else _generation_config.timeout_m * 60 * 1000
            if _generation_config.timeout_m
            else None,
            model=model,
            messages=messages,
            response_schema=response_schema,
            generation_config=generation_config,
            tools=tools,
        )

    @abc.abstractmethod
    async def create_generation_async[T = WithoutStructuredOutput](
        self,
        *,
        model: str | ModelKind | None = None,
        messages: Sequence[AssistantMessage | DeveloperMessage | UserMessage],
        response_schema: type[T] | None = None,
        generation_config: GenerationConfig | GenerationConfigDict | None = None,
        tools: Sequence[Tool[Any]] | None = None,
    ) -> Generation[T]:
        """
        Create a generation from a message sequence asynchronously.

        This is the core method that all provider implementations must implement. It sends
        a sequence of messages to the AI model and processes the response according to
        the provider-specific API requirements.

        Args:
            model: The model identifier to use for generation.
            messages: A sequence of structured Message objects to send to the model.
            response_schema: Optional Pydantic model for structured output parsing.
            generation_config: Optional configuration for the generation request.
            tools: Optional sequence of Tool objects for function calling.

        Returns:
            Generation[T]: An Agentle Generation object containing the model's response,
                potentially with structured output if a response_schema was provided.
        """
        ...

    @abc.abstractmethod
    def price_per_million_tokens_input(
        self, model: str, estimate_tokens: int | None = None
    ) -> float:
        """
        Get the price per million tokens for input/prompt tokens.

        Args:
            model: The name or identifier of the language model.
            estimate_tokens: Optional. An estimated number of tokens that might
                be relevant for tiered pricing models where the price varies
                based on usage volume.

        Returns:
            float: The price in USD per million input tokens.
        """
        ...

    @abc.abstractmethod
    def price_per_million_tokens_output(
        self, model: str, estimate_tokens: int | None = None
    ) -> float:
        """
        Get the price per million tokens for output/completion tokens.

        Args:
            model: The name or identifier of the language model.
            estimate_tokens: Optional. An estimated number of tokens that might
                be relevant for tiered pricing models where the price varies
                based on usage volume.

        Returns:
            float: The price in USD per million output tokens.
        """
        ...

    @abc.abstractmethod
    def map_model_kind_to_provider_model(self, model_kind: ModelKind) -> str:
        """
        Map a model kind to a provider-specific model identifier.

        Args:
            model_kind: The kind of model to map.

        Returns:
            str: The provider-specific model identifier.
        """
        ...

    def _normalize_generation_config(
        self, generation_config: GenerationConfig | GenerationConfigDict | None
    ) -> GenerationConfig:
        if isinstance(generation_config, dict):
            return GenerationConfig.model_validate(generation_config)
        if generation_config is None:
            return GenerationConfig()

        return generation_config

    def _raise_unsuported_model_kind(self, model_kind: ModelKind) -> Never:
        raise NotImplementedError(
            f"Model kind {model_kind} is not supported by {self.__class__.__name__}"
        )

    def __add__(self, other: GenerationProvider) -> FailoverGenerationProvider:
        from agentle.generations.providers.failover.failover_generation_provider import (
            FailoverGenerationProvider,
        )

        return FailoverGenerationProvider(
            generation_providers=[self, other],
            tracing_client=self.tracing_client.unwrap()
            or other.tracing_client.unwrap(),
        )
