"""
Cerebras AI provider implementation for the Agentle framework.

This module provides the CerebrasGenerationProvider class, which enables Agentle
to interact with Cerebras AI models through a consistent interface. It handles all
the provider-specific details of communicating with Cerebras's API while maintaining
compatibility with Agentle's abstraction layer.

The provider supports:
- API key authentication
- Message-based interactions with Cerebras models
- Structured output parsing via response schemas
- Custom HTTP client configuration
- Usage statistics tracking

This implementation transforms Agentle's unified message format into Cerebras's
request format and adapts responses back into Agentle's Generation objects,
providing a consistent experience regardless of the AI provider being used.
"""

from collections.abc import Mapping, Sequence
import logging
from typing import TYPE_CHECKING, Any, cast, override

import httpx
from rsb.adapters.adapter import Adapter
from rsb.contracts.maybe_protocol import MaybeProtocol

# idk why mypy is not recognising this as a module
from agentle.generations.json.json_schema_builder import (  # type: ignore[attr-defined]
    JsonSchemaBuilder,
)
from agentle.generations.models.generation.generation import Generation
from agentle.generations.models.generation.generation_config import GenerationConfig
from agentle.generations.models.generation.generation_config_dict import (
    GenerationConfigDict,
)
from agentle.generations.models.messages.assistant_message import AssistantMessage
from agentle.generations.models.messages.developer_message import DeveloperMessage
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.providers.base.generation_provider import (
    GenerationProvider,
)
from agentle.generations.providers.cerebras._adapters.agentle_message_to_cerebras_message_adapter import (
    AgentleMessageToCerebrasMessageAdapter,
)
from agentle.generations.providers.cerebras._adapters.agentle_tool_to_cerebras_tool_adapter import (
    AgentleToolToCerebrasToolAdapter,
)
from agentle.generations.providers.cerebras._adapters.completion_to_generation_adapter import (
    CerebrasCompletionToGenerationAdapter,
)
from agentle.generations.providers.decorators.model_kind_mapper import (
    override_model_kind,
)
from agentle.generations.providers.types.model_kind import ModelKind
from agentle.generations.tools.tool import Tool
from agentle.generations.tracing.contracts.stateful_observability_client import (
    StatefulObservabilityClient,
)
from agentle.generations.tracing.decorators.observe import observe

if TYPE_CHECKING:
    from cerebras.cloud.sdk.types.chat.completion_create_params import (
        MessageAssistantMessageRequestTyped,
        MessageSystemMessageRequestTyped,
        MessageUserMessageRequestTyped,
    )

logger = logging.getLogger(__name__)
type WithoutStructuredOutput = None


class CerebrasGenerationProvider(GenerationProvider):
    """
    Provider implementation for Cerebras AI services.

    This class implements the GenerationProvider interface for Cerebras AI models,
    allowing seamless integration with the Agentle framework. It handles the conversion
    of Agentle messages to Cerebras format, manages API communication, and processes
    responses back into the standardized Agentle format.

    The provider supports API key authentication, custom HTTP configuration, and
    structured output parsing via response schemas.

    Attributes:
        tracing_client: Optional client for observability and tracing of generation
            requests and responses.
        api_key: Optional API key for authentication with Cerebras AI.
        base_url: Optional custom base URL for the Cerebras API.
        timeout: Optional timeout for API requests.
        max_retries: Maximum number of retries for failed requests.
        default_headers: Optional default HTTP headers for requests.
        default_query: Optional default query parameters for requests.
        http_client: Optional custom HTTP client for requests.
        _strict_response_validation: Whether to enable strict validation of responses.
        warm_tcp_connection: Whether to keep the TCP connection warm.
        message_adapter: Adapter to convert Agentle messages to Cerebras format.
    """

    tracing_client: MaybeProtocol[StatefulObservabilityClient]
    api_key: str | None
    base_url: str | httpx.URL | None
    max_retries: int
    default_headers: Mapping[str, str] | None
    default_query: Mapping[str, object] | None
    http_client: httpx.AsyncClient | None
    _strict_response_validation: bool
    warm_tcp_connection: bool
    message_adapter: Adapter[
        AssistantMessage | UserMessage | DeveloperMessage,
        "MessageSystemMessageRequestTyped | MessageAssistantMessageRequestTyped | MessageUserMessageRequestTyped",
    ]

    def __init__(
        self,
        *,
        tracing_client: StatefulObservabilityClient | None = None,
        api_key: str | None = None,
        base_url: str | httpx.URL | None = None,
        max_retries: int = 2,
        default_headers: Mapping[str, str] | None = None,
        default_query: Mapping[str, object] | None = None,
        http_client: httpx.AsyncClient | None = None,
        _strict_response_validation: bool = False,
        warm_tcp_connection: bool = True,
        message_adapter: Adapter[
            AssistantMessage | UserMessage | DeveloperMessage,
            "MessageSystemMessageRequestTyped | MessageAssistantMessageRequestTyped | MessageUserMessageRequestTyped",
        ]
        | None = None,
    ):
        """
        Initialize the Cerebras Generation Provider.

        Args:
            tracing_client: Optional client for observability and tracing of generation
                requests and responses.
            api_key: Optional API key for authentication with Cerebras AI.
            base_url: Optional custom base URL for the Cerebras API.
            timeout: Optional timeout for API requests.
            max_retries: Maximum number of retries for failed requests.
            default_headers: Optional default HTTP headers for requests.
            default_query: Optional default query parameters for requests.
            http_client: Optional custom HTTP client for requests.
            _strict_response_validation: Whether to enable strict validation of responses.
            warm_tcp_connection: Whether to keep the TCP connection warm.
            message_adapter: Optional adapter to convert Agentle messages to Cerebras format.
        """
        super().__init__(tracing_client=tracing_client)
        self.api_key = api_key
        self.base_url = base_url
        self.max_retries = max_retries
        self.default_headers = default_headers
        self.default_query = default_query
        self.http_client = http_client
        self._strict_response_validation = _strict_response_validation
        self.warm_tcp_connection = warm_tcp_connection
        self.message_adapter = (
            message_adapter or AgentleMessageToCerebrasMessageAdapter()
        )

    @property
    @override
    def organization(self) -> str:
        """
        Get the provider organization identifier.

        Returns:
            str: The organization identifier, which is "cerebras" for this provider.
        """
        return "cerebras"

    @property
    @override
    def default_model(self) -> str:
        """
        The default model to use for generation.
        """
        return "llama-3.3-70b"

    @override
    @observe
    @override_model_kind
    async def create_generation_async[T = WithoutStructuredOutput](
        self,
        *,
        model: str | ModelKind | None = None,
        messages: Sequence[AssistantMessage | DeveloperMessage | UserMessage],
        response_schema: type[T] | None = None,
        generation_config: GenerationConfig | GenerationConfigDict | None = None,
        tools: Sequence[Tool] | None = None,
    ) -> Generation[T]:
        """
        Create a generation asynchronously using a Cerebras AI model.

        This method handles the conversion of Agentle messages to Cerebras's format,
        sends the request to Cerebras's API, and processes the response into Agentle's
        standardized Generation format.

        Args:
            model: The Cerebras model identifier to use for generation.
            messages: A sequence of Agentle messages to send to the model.
            response_schema: Optional Pydantic model for structured output parsing.
            generation_config: Optional configuration for the generation request.
            tools: Optional sequence of Tool objects for function calling (not yet
                supported by Cerebras).

        Returns:
            Generation[T]: An Agentle Generation object containing the model's response,
                potentially with structured output if a response_schema was provided.

        Note:
            Tool/function calling support may vary depending on the Cerebras model
            capabilities. Check the Cerebras documentation for details on supported features.
        """
        from cerebras.cloud.sdk import AsyncCerebras
        from cerebras.cloud.sdk.types.chat.chat_completion import ChatCompletionResponse

        _generation_config = self._normalize_generation_config(generation_config)
        client = AsyncCerebras(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=_generation_config.timeout
            if _generation_config.timeout
            else _generation_config.timeout_s * 1000
            if _generation_config.timeout_s
            else _generation_config.timeout_m * 60 * 1000
            if _generation_config.timeout_m
            else None,
            max_retries=self.max_retries,
            default_headers=self.default_headers,
            default_query=self.default_query,
            http_client=self.http_client,
            _strict_response_validation=self._strict_response_validation,
            warm_tcp_connection=self.warm_tcp_connection,
        )

        tool_adapter = AgentleToolToCerebrasToolAdapter()

        cerebras_completion = cast(
            ChatCompletionResponse,
            await client.chat.completions.create(
                messages=[self.message_adapter.adapt(message) for message in messages],
                model=model or self.default_model,
                tools=[tool_adapter.adapt(tool) for tool in tools] if tools else None,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "json_schema",
                        "strict": True,
                        "schema": JsonSchemaBuilder(
                            cast(type[Any], response_schema),
                            use_defs_instead_of_definitions=True,
                        ).build(dereference=True),
                    },
                }
                if bool(response_schema)
                else None,
                stream=False,
            ),
        )

        return CerebrasCompletionToGenerationAdapter[T](
            response_schema=response_schema,
            model=model or self.default_model,
        ).adapt(cerebras_completion)

    @override
    def map_model_kind_to_provider_model(
        self,
        model_kind: ModelKind,
    ) -> str:
        if model_kind == "category_vision":
            self._raise_unsuported_model_kind(model_kind)

        mapping: Mapping[ModelKind, str] = {
            "category_nano": "llama3.1-8b",
            "category_mini": "qwen-3-32b",
            "category_standard": "llama-3.3-70b",
            "category_pro": "deepseek-r1-distill-llama-70b",
            "category_flagship": "deepseek-r1-distill-llama-70b",
            "category_reasoning": "deepseek-r1-distill-llama-70b",
            "category_coding": "qwen-3-32b",
            "category_instruct": "llama-4-scout-17b-16e-instruct",
            # Experimental fallback to stable
            "category_nano_experimental": "llama3.1-8b",
            "category_mini_experimental": "qwen-3-32b",
            "category_standard_experimental": "llama-4-scout-17b-16e-instruct",
            "category_pro_experimental": "deepseek-r1-distill-llama-70b",
            "category_flagship_experimental": "deepseek-r1-distill-llama-70b",
            "category_reasoning_experimental": "deepseek-r1-distill-llama-70b",
            "category_coding_experimental": "qwen-3-32b",
            "category_instruct_experimental": "llama-4-scout-17b-16e-instruct",
        }

        return mapping[model_kind]

    @override
    def price_per_million_tokens_input(
        self, model: str, estimate_tokens: int | None = None
    ) -> float:
        """
        Get the price per million tokens for input/prompt tokens.

        Args:
            model: The model identifier.
            estimate_tokens: Optional estimate of token count (not used for Cerebras pricing).

        Returns:
            float: The price per million input tokens for the specified model.
        """
        # Pricing data from Cerebras Inference Exploration Tier (search result [6])
        input_prices = {
            "llama3.1-8b": 0.10,  # Llama 3.1 8B [6]
            "qwen-3-32b": 0.40,  # Qwen 3 32B [5][6]
            "deepseek-r1-distill-llama-70b": 2.20,  # Deepseek R1 Distill Llama 70B [6]
            "llama-4-scout-17b-16e-instruct": 0.65,  # Llama 4 Scout [6],
            "llama-3.3-70b": 0.85,  # Llama 3.3 70B [6]
        }

        price = input_prices.get(model)
        if price is None:
            logger.warning(
                f"Cerebras model {model} not found in pricing table. Returning 0.0"
            )
            return 0.0
        return price

    @override
    def price_per_million_tokens_output(
        self, model: str, estimate_tokens: int | None = None
    ) -> float:
        """
        Get the price per million tokens for output/completion tokens.

        Args:
            model: The model identifier.
            estimate_tokens: Optional estimate of token count (not used for Cerebras pricing).

        Returns:
            float: The price per million output tokens for the specified model.
        """
        # Pricing data from Cerebras Inference Exploration Tier (search result [6])
        output_prices = {
            "llama3.1-8b": 0.10,  # Llama 3.1 8B [6]
            "qwen-3-32b": 0.80,  # Qwen 3 32B [5][6]
            "deepseek-r1-distill-llama-70b": 2.50,  # Deepseek R1 Distill Llama 70B [6]
            "llama-4-scout-17b-16e-instruct": 0.85,  # Llama 4 Scout [6]
            "llama-3.3-70b": 1.20,  # Llama 3.3 70B [6]
        }

        price = output_prices.get(model)
        if price is None:
            logger.warning(
                f"Cerebras model {model} not found in pricing table. Returning 0.0"
            )
            return 0.0
        return price
