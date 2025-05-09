from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, Sequence, cast, override

import httpx

from agentle.generations.models.generation.generation import Generation
from agentle.generations.models.generation.generation_config import GenerationConfig
from agentle.generations.models.messages.assistant_message import AssistantMessage
from agentle.generations.models.messages.developer_message import DeveloperMessage
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.providers.base.generation_provider import GenerationProvider
from agentle.generations.providers.openai.adapters.agentle_message_to_openai_message_adapter import (
    AgentleMessageToOpenaiMessageAdapter,
)
from agentle.generations.providers.openai.adapters.chat_completion_to_generation_adapter import (
    ChatCompletionToGenerationAdapter,
)
from agentle.generations.tools.tool import Tool

type WithoutStructuredOutput = None


if TYPE_CHECKING:
    from openai._types import NotGiven


class NotGivenSentinel:
    def __bool__(self) -> Literal[False]:
        return False


NOT_GIVEN = NotGivenSentinel()


class OpenaiGenerationProvider(GenerationProvider):
    """
    OpenAI generation provider.
    """

    api_key: str | None
    organization_name: str | None
    project_name: str | None
    base_url: str | httpx.URL | None
    websocket_base_url: str | httpx.URL | None
    timeout: float | httpx.Timeout | None | NotGiven
    max_retries: int
    default_headers: Mapping[str, str] | None
    default_query: Mapping[str, object] | None
    http_client: httpx.AsyncClient | None

    def __init__(
        self,
        api_key: str,
        *,
        organization_name: str | None = None,
        project_name: str | None = None,
        base_url: str | httpx.URL | None = None,
        websocket_base_url: str | httpx.URL | None = None,
        timeout: float | httpx.Timeout | None | NotGiven | NotGivenSentinel = NOT_GIVEN,
        max_retries: int = 2,
        default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        from openai._types import NOT_GIVEN as OPENAI_NOT_GIVEN

        if timeout is NOT_GIVEN:
            timeout = OPENAI_NOT_GIVEN

        self.api_key = api_key
        self.organization_name = organization_name
        self.project_name = project_name
        self.base_url = base_url
        self.websocket_base_url = websocket_base_url
        self.timeout = cast(float | httpx.Timeout | None | NotGiven, timeout)
        self.max_retries = max_retries
        self.default_headers = default_headers
        self.default_query = default_query
        self.http_client = http_client

    @override
    async def create_generation_async[T = WithoutStructuredOutput](
        self,
        *,
        model: str | None = None,
        messages: Sequence[AssistantMessage | DeveloperMessage | UserMessage],
        response_schema: type[T] | None = None,
        generation_config: GenerationConfig | None = None,
        tools: Sequence[Tool[Any]] | None = None,
    ) -> Generation[T]:
        from openai import AsyncOpenAI
        from openai.types.chat.chat_completion import ChatCompletion

        client = AsyncOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            websocket_base_url=self.websocket_base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
            default_headers=self.default_headers,
            default_query=self.default_query,
            http_client=self.http_client,
            organization=self.organization_name,
            project=self.project_name,
        )

        input_message_adapter = AgentleMessageToOpenaiMessageAdapter()

        chat_completion: ChatCompletion = await client.chat.completions.create(  # type: ignore
            messages=[input_message_adapter.adapt(message) for message in messages],
            model=model or self.default_model,
        )

        output_adapter = ChatCompletionToGenerationAdapter[T](
            response_schema=response_schema
        )

        return output_adapter.adapt(chat_completion)

        raise NotImplementedError

    @property
    @override
    def default_model(self) -> str:
        return "gpt-4o"

    @override
    def price_per_million_tokens_input(
        self, model: str, estimate_tokens: int | None = None
    ) -> float:
        """
        Get the price per million tokens for input/prompt tokens.
        """
        return 0.0

    @override
    def price_per_million_tokens_output(
        self, model: str, estimate_tokens: int | None = None
    ) -> float:
        """
        Get the price per million tokens for output/completion tokens.
        """
        return 0.0
