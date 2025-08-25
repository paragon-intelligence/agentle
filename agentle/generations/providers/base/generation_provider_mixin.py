"""
Abstract base class defining the contract for AI generation providers in Agentle.

This module defines the GenerationProviderType abstract base class, which serves as the
foundation for all AI provider implementations in the Agentle framework. It establishes
a common interface that all providers must implement, ensuring consistency across
different AI services.

The GenerationProviderType abstract class defines methods for generating AI completions
from both prompts and message sequences, supporting both synchronous and asynchronous
execution patterns. It also includes support for structured output parsing through
generic type parameters and tool/function calling capabilities.

Provider implementations (such as for OpenAI, Google, Anthropic, etc.) inherit from
this base class and implement the abstract methods according to each provider's specific
API requirements, while maintaining the common interface for framework consumers.
"""

from __future__ import annotations

import abc
from collections.abc import MutableSequence, Sequence
from typing import TYPE_CHECKING, Any, Never, cast

from rsb.coroutines.run_sync import run_sync
from rsb.models.base_model import BaseModel
from rsb.models.field import Field

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
from agentle.generations.tools.tool_execution_result import ToolExecutionResult
from agentle.prompts.models.prompt import Prompt
from agentle.generations.tracing.otel_client import OtelClient

type WithoutStructuredOutput = None

if TYPE_CHECKING:
    from agentle.generations.providers.failover.failover_generation_provider import (
        FailoverGenerationProvider,
    )


class GenerationProviderMixin(BaseModel, abc.ABC):
    otel_clients: Sequence[OtelClient] | OtelClient = Field(default_factory=list)

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

    def generate_by_prompt[T = WithoutStructuredOutput](
        self,
        prompt: str
        | Prompt
        | Part
        | MutableSequence[
            TextPart
            | FilePart
            | Tool[Any]
            | ToolExecutionSuggestion
            | ToolExecutionResult
        ],
        *,
        model: str | ModelKind | None = None,
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
            self.generate_by_prompt_async,
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

    async def generate_by_prompt_async[T = WithoutStructuredOutput](
        self,
        prompt: str
        | Prompt
        | Part
        | MutableSequence[
            TextPart
            | FilePart
            | Tool[Any]
            | ToolExecutionSuggestion
            | ToolExecutionResult
        ],
        *,
        model: str | ModelKind | None = None,
        developer_prompt: str | Prompt | None = None,
        response_schema: type[T] | None = None,
        generation_config: GenerationConfig | GenerationConfigDict | None = None,
        tools: Sequence[Tool] | None = None,
    ) -> Generation[T]:
        """
        Create a generation from a prompt asynchronously.

        This method converts various prompt formats into a structured message sequence
        and calls the generate_async method with the converted messages.

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
        user_message_parts: MutableSequence[
            TextPart
            | FilePart
            | Tool[Any]
            | ToolExecutionSuggestion
            | ToolExecutionResult
        ]
        match prompt:
            case str():
                user_message_parts = cast(
                    MutableSequence[
                        TextPart
                        | FilePart
                        | Tool[Any]
                        | ToolExecutionSuggestion
                        | ToolExecutionResult
                    ],
                    [TextPart(text=prompt)],
                )
            case Prompt():
                user_message_parts = cast(
                    MutableSequence[
                        TextPart
                        | FilePart
                        | Tool[Any]
                        | ToolExecutionSuggestion
                        | ToolExecutionResult
                    ],
                    [TextPart(text=prompt.content)],
                )
            case TextPart() | FilePart() | Tool() | ToolExecutionSuggestion():
                user_message_parts = cast(
                    MutableSequence[
                        TextPart
                        | FilePart
                        | Tool[Any]
                        | ToolExecutionSuggestion
                        | ToolExecutionResult
                    ],
                    [prompt],
                )
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

        return await self.generate_async(
            model=model,
            messages=[developer_message, user_message],
            response_schema=response_schema,
            generation_config=generation_config,
            tools=tools,
        )

    def generate[T = WithoutStructuredOutput](
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
            self.generate_async,
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
    async def generate_async[T = WithoutStructuredOutput](
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

    def _resolve_model(self, model: str | ModelKind | None) -> str:
        if model and model in [
            "category_nano",
            "category_mini",
            "category_standard",
            "category_pro",
            "category_flagship",
            "category_reasoning",
            "category_vision",
            "category_coding",
            "category_instruct",
            "category_nano_experimental",
            "category_mini_experimental",
            "category_standard_experimental",
            "category_pro_experimental",
            "category_flagship_experimental",
            "category_reasoning_experimental",
            "category_vision_experimental",
            "category_coding_experimental",
            "category_instruct_experimental",
        ]:
            return self.map_model_kind_to_provider_model(
                model_kind=cast(ModelKind, model)
            )

        return model or self.default_model

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

    def __add__(
        self, other: GenerationProviderMixin | Sequence[GenerationProviderMixin]
    ) -> FailoverGenerationProvider:
        from agentle.generations.providers.failover.failover_generation_provider import (
            FailoverGenerationProvider,
        )

        match other:
            case GenerationProviderMixin():
                providers: MutableSequence[GenerationProviderMixin] = [self, other]
            case _:
                providers = cast(list[GenerationProviderMixin], [self] + list(other))

        return FailoverGenerationProvider(
            generation_providers=providers,
            otel_clients=self.otel_clients or other.otel_clients
            if isinstance(other, GenerationProviderMixin)
            else other[0].otel_clients,
        )
