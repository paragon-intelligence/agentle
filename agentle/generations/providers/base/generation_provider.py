import abc
from collections.abc import Sequence
from typing import cast

from rsb.containers.maybe import Maybe
from rsb.contracts.maybe_protocol import MaybeProtocol
from rsb.coroutines.run_sync import run_sync

from agentle.generations.models.generation.generation import Generation
from agentle.generations.models.generation.generation_config import GenerationConfig
from agentle.generations.models.message_parts.file import FilePart
from agentle.generations.models.message_parts.part import Part
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.message_parts.tool_execution_suggestion import (
    ToolExecutionSuggestion,
)
from agentle.generations.models.messages.developer_message import DeveloperMessage
from agentle.generations.models.messages.message import Message
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.tools.tool import Tool
from agentle.generations.tracing.contracts.stateful_observability_client import (
    StatefulObservabilityClient,
)
from agentle.prompts.models.prompt import Prompt

type WithoutStructuredOutput = None


class GenerationProvider(abc.ABC):
    tracing_client: MaybeProtocol[StatefulObservabilityClient]

    def __init__(
        self,
        tracing_client: StatefulObservabilityClient | None = None,
    ) -> None:
        self.tracing_client = Maybe(tracing_client)

    @property
    @abc.abstractmethod
    def organization(self) -> str: ...

    def create_generation_by_prompt[T = WithoutStructuredOutput](
        self,
        *,
        model: str,
        prompt: str | Prompt | Part | Sequence[Part],
        developer_prompt: str | Prompt,
        response_schema: type[T] | None = None,
        generation_config: GenerationConfig | None = None,
        tools: Sequence[Tool] | None = None,
    ) -> Generation[T]:
        return run_sync(
            self.create_generation_by_prompt_async,
            timeout=generation_config.timeout if generation_config else None,
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
        model: str,
        prompt: str | Prompt | Part | Sequence[Part],
        developer_prompt: str | Prompt,
        response_schema: type[T] | None = None,
        generation_config: GenerationConfig | None = None,
        tools: Sequence[Tool] | None = None,
    ) -> Generation[T]:
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
        model: str,
        messages: Sequence[Message],
        response_schema: type[T] | None = None,
        generation_config: GenerationConfig | None = None,
        tools: Sequence[Tool] | None = None,
    ) -> Generation[T]:
        return run_sync(
            self.create_generation_async,
            timeout=generation_config.timeout if generation_config else None,
            model=model,
            messages=messages,
            response_schema=response_schema,
            generation_config=generation_config,
        )

    @abc.abstractmethod
    async def create_generation_async[T = WithoutStructuredOutput](
        self,
        *,
        model: str,
        messages: Sequence[Message],
        response_schema: type[T] | None = None,
        generation_config: GenerationConfig | None = None,
        tools: Sequence[Tool] | None = None,
    ) -> Generation[T]: ...
