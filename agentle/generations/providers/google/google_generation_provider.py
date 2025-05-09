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

import logging
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast, override

from rsb.adapters.adapter import Adapter

from agentle.generations.models.generation.generation import Generation
from agentle.generations.models.generation.generation_config import GenerationConfig
from agentle.generations.models.messages.assistant_message import AssistantMessage
from agentle.generations.models.messages.developer_message import DeveloperMessage
from agentle.generations.models.messages.message import Message
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.providers.base.generation_provider import (
    GenerationProvider,
)
from agentle.generations.providers.google.adapters.agentle_tool_to_google_tool_adapter import (
    AgentleToolToGoogleToolAdapter,
)
from agentle.generations.providers.google.adapters.generate_generate_content_response_to_generation_adapter import (
    GenerateGenerateContentResponseToGenerationAdapter,
)
from agentle.generations.providers.google.adapters.message_to_google_content_adapter import (
    MessageToGoogleContentAdapter,
)
from agentle.generations.providers.google.function_calling_config import (
    FunctionCallingConfig,
)
from agentle.generations.tools.tool import Tool
from agentle.generations.tracing.contracts.stateful_observability_client import (
    StatefulObservabilityClient,
)
from agentle.generations.tracing.tracing_manager import TracingManager

if TYPE_CHECKING:
    from google.auth.credentials import Credentials
    from google.genai.client import (
        DebugConfig,
        HttpOptions,
    )
    from google.genai.types import Content


type WithoutStructuredOutput = None

logger = logging.getLogger(__name__)


class GoogleGenerationProvider(GenerationProvider):
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
        tracing_manager: Manager for tracing generation activities.
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
    tracing_manager: TracingManager

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
        self.tracing_manager = TracingManager(
            tracing_client=tracing_client,
            provider=self,
        )

    @property
    @override
    def default_model(self) -> str:
        """
        The default model to use for generation.
        """
        return "gemini-2.0-flash"

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
        model: str | None = None,
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
        used_model = model or self.default_model
        _generation_config = generation_config or GenerationConfig()
        is_final_generation = response_schema is not None

        # Prepare input data for tracing
        input_data: dict[str, Any] = {
            "messages": [
                {
                    "role": msg.role,
                    "content": "".join(str(part) for part in msg.parts),
                }
                for msg in messages
            ],
            "response_schema": str(response_schema) if response_schema else None,
            "temperature": _generation_config.temperature,
            "top_p": _generation_config.top_p,
            "top_k": _generation_config.top_k,
            "max_output_tokens": _generation_config.max_output_tokens,
            "tools_count": len(tools) if tools else 0,
            "message_count": len(messages),
            "has_tools": tools is not None and len(tools) > 0,
            "has_schema": response_schema is not None,
        }

        # Set up tracing using the tracing manager
        trace_client, generation_client = await self.tracing_manager.setup_trace(
            generation_config=_generation_config,
            model=used_model,
            input_data=input_data,
            is_final_generation=is_final_generation,
        )

        # Extract trace metadata if available
        trace_metadata: dict[str, Any] = {}
        trace_params = _generation_config.trace_params
        if "metadata" in trace_params:
            metadata_val = trace_params["metadata"]
            if isinstance(metadata_val, dict):
                # Convert to properly typed dict
                for k, v in metadata_val.items():
                    if isinstance(k, str):
                        trace_metadata[k] = v

        try:
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
                tools=[
                    AgentleToolToGoogleToolAdapter().adapt(tool) for tool in final_tools
                ]
                if final_tools
                else None,
                max_output_tokens=_generation_config.max_output_tokens,
                response_schema=response_schema if bool(response_schema) else None,
                response_mime_type="application/json"
                if bool(response_schema)
                else None,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=disable_function_calling,
                    maximum_remote_calls=maximum_remote_calls,
                    ignore_call_history=ignore_call_history,
                ),
            )

            contents = [self.message_adapter.adapt(message) for message in messages]
            generate_content_response = await client.aio.models.generate_content(
                model=used_model,
                contents=cast(types.ContentListUnion, contents),
                config=config,
            )

            # Create the response
            response = GenerateGenerateContentResponseToGenerationAdapter[T](
                response_schema=response_schema,
                model=used_model,
            ).adapt(generate_content_response)

            # Prepare output data for tracing
            output_data: dict[str, Any] = {
                "completion": response.text,
                "usage": {
                    "input_tokens": response.usage.prompt_tokens
                    if response.usage
                    else None,
                    "output_tokens": response.usage.completion_tokens
                    if response.usage
                    else None,
                    "total_tokens": response.usage.total_tokens
                    if response.usage
                    else None,
                },
            }

            input_cost = (
                self.price_per_million_tokens_input(
                    used_model, response.usage.prompt_tokens
                )
                if response.usage.prompt_tokens
                else 0.0
            )

            output_cost = (
                self.price_per_million_tokens_output(
                    used_model, response.usage.completion_tokens
                )
                if response.usage.completion_tokens
                else 0.0
            )

            total_cost = input_cost + output_cost

            # Add cost information to output_data
            output_data["usage"]["input_cost"] = input_cost
            output_data["usage"]["output_cost"] = output_cost
            output_data["usage"]["total_cost"] = total_cost

            # Add user-specified output if available
            if "output" in trace_params:
                custom_output = trace_params["output"]
                if isinstance(custom_output, dict):
                    for k, v in custom_output.items():
                        output_data[k] = v
                else:
                    output_data["user_defined_output"] = custom_output

            # Complete the generation and trace if needed
            await self.tracing_manager.complete_generation(
                generation_client=generation_client,
                start_time=start,
                output_data=output_data,
                trace_metadata=trace_metadata,
            )

            # If this is the final generation, complete the trace
            if is_final_generation:
                final_output = {
                    "final_response": response.text,
                    "structured_output": response.parsed
                    if hasattr(response, "parsed")
                    else None,
                    "usage": {
                        "input": response.usage.prompt_tokens
                        if response.usage
                        else None,
                        "output": response.usage.completion_tokens
                        if response.usage
                        else None,
                        "total": response.usage.total_tokens
                        if response.usage
                        else None,
                        "unit": "TOKENS",
                        "input_cost": output_data["usage"].get("input_cost"),
                        "output_cost": output_data["usage"].get("output_cost"),
                        "total_cost": output_data["usage"].get("total_cost"),
                        "currency": "USD",
                    },
                }
                await self.tracing_manager.complete_trace(
                    trace_client=trace_client,
                    generation_config=_generation_config,
                    output_data=final_output,
                    success=True,
                )

            return response

        except Exception as e:
            # Handle errors using the tracing manager
            await self.tracing_manager.handle_error(
                generation_client=generation_client,
                trace_client=trace_client,
                generation_config=_generation_config,
                start_time=start,
                error=e,
                trace_metadata=trace_metadata,
            )
            # Re-raise the exception
            raise

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
        if not self.use_vertex_ai:
            return 0.0

        # Pricing in USD per million tokens (from https://cloud.google.com/vertex-ai/generative-ai/pricing)
        # Standard pricing for most models
        model_to_price_per_million: Mapping[str, float | tuple[float, float, int]] = {
            # Gemini 2.5 models with tiered pricing: (price_low_tier, price_high_tier, threshold)
            "gemini-2.5-pro": (1.25, 2.5, 200_000),  # <= 200K: $1.25, > 200K: $2.5
            "gemini-2.5-flash": (0.15, 0.15, 200_000),  # Same price for both tiers
            # Gemini models
            "gemini-1.0-pro": 3.50,
            "gemini-1.0-pro-vision": 3.50,
            "gemini-1.5-flash": 0.35,
            "gemini-1.5-pro": 7.00,
            "gemini-2.0-flash": 0.40,
            "gemini-2.0-pro": 6.00,
            # Claude models (Anthropic)
            "claude-3-5-sonnet": 3.00,
            "claude-3-5-sonnet-v2": 3.00,
            "claude-3-5-haiku": 0.80,
            "claude-3-7-sonnet": 3.00,
            "claude-3-haiku": 0.25,
            "claude-3-opus": 15.00,
            # Llama models (Meta)
            "llama-3-1-405b": 5.00,
            "llama-3-3-70b": 0.72,
            "llama-4-scout": 0.25,
            "llama-4-maverick": 0.35,
            # Mistral models
            "mistral-small-3-1": 0.10,
            "mistral-large-24-11": 2.00,
            "mistral-nemo": 0.15,
            "codestral-25-01": 0.30,
            # AI21 models
            "jamba-1-5-large": 2.00,
            "jamba-1-5-mini": 0.20,
        }

        price_info = model_to_price_per_million.get(model)
        if price_info is None:
            logger.warning(
                f"Model {model} not found in model_to_price_per_million yet. Returning 0.0 to not raise any errors."
            )
            return 0.0

        # If estimate_tokens is None, return the base price (or lower tier price for tiered models)
        if estimate_tokens is None:
            if isinstance(price_info, tuple):
                # Return the lower tier price for tiered models
                return price_info[0]
            return price_info

        # Calculate the price based on token tiers if applicable
        if isinstance(price_info, tuple):
            low_tier_price, high_tier_price, threshold = price_info

            # If tokens exceed threshold, use the higher tier price
            if estimate_tokens > threshold:
                return high_tier_price * (estimate_tokens / 1_000_000)
            else:
                return low_tier_price * (estimate_tokens / 1_000_000)
        else:
            # Standard pricing for non-tiered models
            return price_info * (estimate_tokens / 1_000_000)

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
        if not self.use_vertex_ai:
            return 0.0

        # Pricing in USD per million tokens (from https://cloud.google.com/vertex-ai/generative-ai/pricing)
        # Standard pricing for most models
        model_to_price_per_million: Mapping[str, float | tuple[float, float, int]] = {
            # Gemini 2.5 models with tiered pricing: (price_low_tier, price_high_tier, threshold)
            "gemini-2.5-pro": (10.0, 15.0, 200_000),  # <= 200K: $10, > 200K: $15
            "gemini-2.5-flash": (0.60, 0.60, 200_000),  # Same price for both tiers
            # Gemini models
            "gemini-1.0-pro": 10.50,
            "gemini-1.0-pro-vision": 10.50,
            "gemini-1.5-flash": 1.05,
            "gemini-1.5-pro": 21.00,
            "gemini-2.0-flash": 1.20,
            "gemini-2.0-pro": 18.00,
            # Claude models (Anthropic)
            "claude-3-5-sonnet": 15.00,
            "claude-3-5-sonnet-v2": 15.00,
            "claude-3-5-haiku": 4.00,
            "claude-3-7-sonnet": 15.00,
            "claude-3-haiku": 1.25,
            "claude-3-opus": 75.00,
            # Llama models (Meta)
            "llama-3-1-405b": 16.00,
            "llama-3-3-70b": 0.72,
            "llama-4-scout": 0.70,
            "llama-4-maverick": 1.15,
            # Mistral models
            "mistral-small-3-1": 0.30,
            "mistral-large-24-11": 6.00,
            "mistral-nemo": 0.15,
            "codestral-25-01": 0.90,
            # AI21 models
            "jamba-1-5-large": 8.00,
            "jamba-1-5-mini": 0.40,
        }

        price_info = model_to_price_per_million.get(model)
        if price_info is None:
            logger.warning(
                f"Model {model} not found in model_to_price_per_million yet. Returning 0.0 to not raise any errors."
            )
            return 0.0

        # If estimate_tokens is None, return the base price (or lower tier price for tiered models)
        if estimate_tokens is None:
            if isinstance(price_info, tuple):
                # Return the lower tier price for tiered models
                return price_info[0]
            return price_info

        # For tiered models
        if isinstance(price_info, tuple):
            # Note: According to docs, if input exceeds threshold, both input AND output
            # are charged at higher tier. Without knowing input tokens here, we can't
            # determine this accurately. For now, use the lower tier price.
            low_tier_price = price_info[0]
            return low_tier_price * (estimate_tokens / 1_000_000)
        else:
            # Standard pricing for non-tiered models
            return price_info * (estimate_tokens / 1_000_000)
