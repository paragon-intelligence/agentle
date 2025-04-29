"""
Google AI provider implementation for the Agentle framework.

This module provides integration with Google's Generative AI services, allowing
Agentle to use models from the Google AI ecosystem. It implements the necessary
provider interfaces to maintain compatibility with the broader Agentle framework
while handling all Google-specific implementation details internally.

The module supports:
- Both API key and credential-based authentication
- Optional Vertex AI integration for enterprise deployments
- Configurable HTTP options and timeouts
- Function/tool calling capabilities
- Structured output parsing via response schemas
- Tracing and observability integration

This provider transforms Agentle's unified message format into Google's Content
format and adapts responses back into Agentle's Generation objects, maintaining
a consistent interface regardless of the underlying AI provider being used.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING, cast, override

from rsb.adapters.adapter import Adapter

from agentle.generations.models.generation.generation import Generation
from agentle.generations.models.generation.generation_config import GenerationConfig
from agentle.generations.models.messages.assistant_message import AssistantMessage
from agentle.generations.models.messages.developer_message import DeveloperMessage
from agentle.generations.models.messages.message import Message
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.pricing.price_retrievable import PriceRetrievable
from agentle.generations.providers.base.generation_provider import (
    GenerationProvider,
)
from agentle.generations.providers.google._adapters.agentle_tool_to_google_tool_adapter import (
    AgentleToolToGoogleToolAdapter,
)
from agentle.generations.providers.google._adapters.generate_generate_content_response_to_generation_adapter import (
    GenerateGenerateContentResponseToGenerationAdapter,
)
from agentle.generations.providers.google._adapters.message_to_google_content_adapter import (
    MessageToGoogleContentAdapter,
)
from agentle.generations.providers.google.function_calling_config import (
    FunctionCallingConfig,
)
from agentle.generations.tools.tool import Tool
from agentle.generations.tracing.contracts.stateful_observability_client import (
    StatefulObservabilityClient,
)

if TYPE_CHECKING:
    from google.auth.credentials import Credentials
    from google.genai.client import (
        DebugConfig,
        HttpOptions,
    )
    from google.genai.types import Content


type WithoutStructuredOutput = None


class GoogleGenerationProvider(GenerationProvider, PriceRetrievable):
    """
    Provider implementation for Google's Generative AI service.

    This class implements the GenerationProvider interface for Google AI models,
    allowing seamless integration with the Agentle framework. It supports both
    standard API key authentication and Vertex AI integration for enterprise
    deployments.

    The provider handles message format conversion, tool adaptation, function
    calling configuration, and response processing to maintain consistency with
    Agentle's unified interface.

    Attributes:
        use_vertex_ai: Whether to use Google Vertex AI instead of standard API.
        api_key: Optional API key for authentication with Google AI.
        credentials: Optional credentials object for authentication.
        project: Google Cloud project ID (required for Vertex AI).
        location: Google Cloud region (required for Vertex AI).
        debug_config: Optional configuration for debug logging.
        http_options: HTTP options for the Google AI client.
        message_adapter: Adapter to convert Agentle messages to Google Content format.
        function_calling_config: Configuration for function calling behavior.
    """

    use_vertex_ai: bool
    api_key: str | None
    credentials: Credentials | None
    project: str | None
    location: str | None
    debug_config: DebugConfig | None
    http_options: HttpOptions | None
    message_adapter: Adapter[AssistantMessage | UserMessage | DeveloperMessage, Content]
    function_calling_config: FunctionCallingConfig

    def __init__(
        self,
        *,
        tracing_client: StatefulObservabilityClient | None = None,
        use_vertex_ai: bool = False,
        api_key: str | None | None = None,
        credentials: Credentials | None = None,
        project: str | None = None,
        location: str | None = None,
        debug_config: DebugConfig | None = None,
        http_options: HttpOptions | None = None,
        message_adapter: Adapter[
            AssistantMessage | UserMessage | DeveloperMessage, Content
        ]
        | None = None,
        function_calling_config: FunctionCallingConfig | None = None,
    ) -> None:
        """
        Initialize the Google Generation Provider.

        Args:
            tracing_client: Optional client for observability and tracing.
            use_vertex_ai: Whether to use Google Vertex AI instead of standard API.
            api_key: Optional API key for authentication with Google AI.
            credentials: Optional credentials object for authentication.
            project: Google Cloud project ID (required for Vertex AI).
            location: Google Cloud region (required for Vertex AI).
            debug_config: Optional configuration for debug logging.
            http_options: HTTP options for the Google AI client.
            message_adapter: Optional adapter to convert Agentle messages to Google Content.
            function_calling_config: Optional configuration for function calling behavior.
        """
        super().__init__(tracing_client=tracing_client)
        self.use_vertex_ai = use_vertex_ai
        self.api_key = api_key
        self.credentials = credentials
        self.project = project
        self.location = location
        self.debug_config = debug_config
        self.http_options = http_options
        self.message_adapter = message_adapter or MessageToGoogleContentAdapter()
        self.function_calling_config = function_calling_config or {}

    @property
    @override
    def organization(self) -> str:
        """
        Get the provider organization identifier.

        Returns:
            str: The organization identifier, which is "google" for this provider.
        """
        return "google"

    @override
    async def create_generation_async[T = WithoutStructuredOutput](
        self,
        *,
        model: str,
        messages: Sequence[Message],
        response_schema: type[T] | None = None,
        generation_config: GenerationConfig | None = None,
        tools: Sequence[Tool] | None = None,
    ) -> Generation[T]:
        """
        Create a generation asynchronously using a Google AI model.

        This method handles the conversion of Agentle messages and tools to Google's
        format, sends the request to Google's API, and processes the response into
        Agentle's standardized Generation format.

        Args:
            model: The Google AI model identifier to use (e.g., "gemini-1.5-pro").
            messages: A sequence of Agentle messages to send to the model.
            response_schema: Optional Pydantic model for structured output parsing.
            generation_config: Optional configuration for the generation request.
            tools: Optional sequence of Tool objects for function calling.

        Returns:
            Generation[T]: An Agentle Generation object containing the model's response,
                potentially with structured output if a response_schema was provided.
        """
        from google import genai
        from google.genai import types

        start = datetime.now()

        _generation_config = generation_config or GenerationConfig()

        _http_options = self.http_options or types.HttpOptions()
        # change so if the timeout is provided in the constructor and the user doesnt inform the timeout in the generation config, the timeout in the constructor is used
        _http_options.timeout = (
            int(
                _generation_config.timeout * 1000
            )  # Convertendo de segundos para milissegundos
            if _generation_config.timeout
            else _http_options.timeout
        )

        client = genai.Client(
            vertexai=self.use_vertex_ai,
            api_key=self.api_key,
            credentials=self.credentials,
            project=self.project,
            location=self.location,
            debug_config=self.debug_config,
            http_options=_http_options,
        )

        system_instruction: Content | None = None
        first_message = messages[0]
        if isinstance(first_message, DeveloperMessage):
            system_instruction = self.message_adapter.adapt(first_message)

        message_tools = [
            part
            for message in messages
            for part in message.parts
            if isinstance(part, Tool)
        ]

        final_tools = (
            list(tools or []) + message_tools if tools or message_tools else None
        )

        disable_function_calling = self.function_calling_config.get("disable", True)
        # if disable_function_calling is True, set maximum_remote_calls to None
        maximum_remote_calls = None if disable_function_calling else 10
        ignore_call_history = self.function_calling_config.get(
            "ignore_call_history", False
        )

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=_generation_config.temperature,
            top_p=_generation_config.top_p,
            top_k=_generation_config.top_k,
            candidate_count=_generation_config.n,
            tools=[AgentleToolToGoogleToolAdapter().adapt(tool) for tool in final_tools]
            if final_tools
            else None,
            max_output_tokens=_generation_config.max_output_tokens,
            response_schema=response_schema if bool(response_schema) else None,
            response_mime_type="application/json" if bool(response_schema) else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=disable_function_calling,
                maximum_remote_calls=maximum_remote_calls,
                ignore_call_history=ignore_call_history,
            ),
        )

        contents = [self.message_adapter.adapt(message) for message in messages]
        generate_content_response = await client.aio.models.generate_content(
            model=model,
            contents=cast(types.ContentListUnion, contents),
            config=config,
        )

        return GenerateGenerateContentResponseToGenerationAdapter[T](
            response_schema=response_schema, start_time=start, model=model
        ).adapt(generate_content_response)

    @override
    def price_per_million_tokens_input(
        self, model: str, estimate_tokens: int | None = None
    ) -> float:
        """
        Get the price per million tokens for input/prompt tokens.

        Args:
            model: The model identifier.
            estimate_tokens: Optional estimate of token count.

        Returns:
            float: The price per million input tokens for the specified model.
        """
        return 0.0  # TODO(arthur)

    @override
    def price_per_million_tokens_output(
        self, model: str, estimate_tokens: int | None = None
    ) -> float:
        """
        Get the price per million tokens for output/completion tokens.

        Args:
            model: The model identifier.
            estimate_tokens: Optional estimate of token count.

        Returns:
            float: The price per million output tokens for the specified model.
        """
        return 0.0  # TODO(arthur)
