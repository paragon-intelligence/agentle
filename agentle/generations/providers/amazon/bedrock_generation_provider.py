from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Literal, Sequence, cast, override

from rsb.coroutines.run_async import run_async
from rsb.models.field import Field
from rsb.models.private_attr import PrivateAttr

from agentle.generations.collections.message_sequence import MessageSequence
from agentle.generations.models.generation.generation import Generation
from agentle.generations.models.generation.generation_config import GenerationConfig
from agentle.generations.models.generation.generation_config_dict import (
    GenerationConfigDict,
)
from agentle.generations.models.messages.assistant_message import AssistantMessage
from agentle.generations.models.messages.developer_message import DeveloperMessage
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.providers.amazon.adapters.agentle_message_to_boto_message import (
    AgentleMessageToBotoMessage,
)
from agentle.generations.providers.amazon.adapters.agentle_tool_to_bedrock_tool_adapter import (
    AgentleToolToBedrockToolAdapter,
)
from agentle.generations.providers.amazon.adapters.converse_response_to_agentle_generation_adapter import (
    ConverseResponseToAgentleGenerationAdapter,
)
from agentle.generations.providers.amazon.adapters.generation_config_to_inference_config import (
    GenerationConfigToInferenceConfigAdapter,
)
from agentle.generations.providers.amazon.adapters.response_schema_to_bedrock_tool_adapter import (
    ResponseSchemaToBedrockToolAdapter,
)
from agentle.generations.providers.amazon.boto_config import BotoConfig
from agentle.generations.providers.amazon.models.specific_tool import SpecificTool
from agentle.generations.providers.amazon.models.text_content import TextContent
from agentle.generations.providers.amazon.models.tool_choice import ToolChoice
from agentle.generations.providers.amazon.models.tool_config import ToolConfig
from agentle.generations.providers.base.generation_provider_mixin import (
    GenerationProviderMixin,
)
from agentle.generations.providers.types.model_kind import ModelKind
from agentle.generations.tools.tool import Tool
from agentle.generations.tracing.observe import observe

if TYPE_CHECKING:
    from mypy_boto3_bedrock_runtime.client import BedrockRuntimeClient


logger = logging.getLogger(__name__)


class BedrockGenerationProvider(GenerationProviderMixin):
    type: Literal["bedrock"] = Field(default="bedrock")
    region_name: str = Field(default="us-east-1")
    access_key_id: str | None = Field(default=None)
    secret_access_key: str | None = Field(default=None)
    config: BotoConfig | None = Field(default=None)
    _client: BedrockRuntimeClient | None = PrivateAttr(default=None)

    @property
    def client(self) -> BedrockRuntimeClient:
        if self._client is None:
            raise ValueError("Bedrock client not initialized")
        return self._client

    @override
    def model_post_init(self, context: Any, /) -> None:
        super().model_post_init(context)

        import boto3

        self._client = cast(
            BedrockRuntimeClient,
            boto3.client(
                "bedrock-runtime",
                aws_access_key_id=self.access_key_id or os.getenv("AWS_ACCESS_KEY_ID"),
                aws_secret_access_key=self.secret_access_key
                or os.getenv("AWS_SECRET_ACCESS_KEY"),
                region_name=self.region_name,
                config=self.config,
            ),
        )

    @property
    @override
    def default_model(self) -> str:
        return "us.anthropic.claude-sonnet-4-20250514-v1:0"

    @property
    @override
    def organization(self) -> str:
        return "aws"

    @override
    @observe
    async def generate_async[T](
        self,
        *,
        model: str | None | ModelKind = None,
        messages: Sequence[AssistantMessage | DeveloperMessage | UserMessage],
        response_schema: type[T] | None = None,
        generation_config: GenerationConfig | GenerationConfigDict | None = None,
        tools: Sequence[Tool[Any]] | None = None,
    ) -> Generation[T]:
        message_adapter = AgentleMessageToBotoMessage()

        message_sequence = MessageSequence(messages)

        messages_without_system = message_sequence.without_developer_prompt().elements

        system_message: DeveloperMessage | None = (
            messages_without_system[0]
            if isinstance(messages_without_system[0], DeveloperMessage)
            else None
        )

        conversation = [
            message_adapter.adapt(message) for message in messages_without_system
        ]

        inference_config_adapter = GenerationConfigToInferenceConfigAdapter()
        tool_adapter = AgentleToolToBedrockToolAdapter()

        _generation_config = self._normalize_generation_config(generation_config)

        if _generation_config.n > 1:
            raise ValueError(
                f"Amazon Bedrock does not directly support 'n' > 1. n = {_generation_config.n}"
            )

        _inference_config = inference_config_adapter.adapt(_generation_config)

        # TODO: rs_tool = response_schema_to_bedrock_tool(response_schema)
        rs_tool = (
            ResponseSchemaToBedrockToolAdapter().adapt(response_schema)
            if response_schema
            else None
        )

        extra_tools = [rs_tool] if rs_tool else []

        _tools = tools or []

        _tool_config = (
            ToolConfig(
                tools=[tool_adapter.adapt(tool) for tool in _tools] + extra_tools,
                toolChoice=ToolChoice(auto={})
                if response_schema is None
                else ToolChoice(tool=SpecificTool(name=response_schema.__name__)),
            )
            if _tools or extra_tools
            else None
        )

        _system = [
            TextContent(
                text=system_message.text
                if system_message
                else "You are a helpful assistant"
            )
        ]

        _model = self._resolve_model(model)

        async with asyncio.timeout(_generation_config.timeout_in_seconds):
            if _tool_config:
                response = await run_async(
                    self.client.converse,
                    modelId=_model,
                    system=_system,  # type: ignore[reportArgumentType]
                    messages=conversation,  # type: ignore[reportArgumentType]
                    inferenceConfig=_inference_config,
                    toolConfig=_tool_config,  # type: ignore[reportArgumentType]
                )

            else:
                response = await run_async(
                    self.client.converse,
                    modelId=_model,
                    system=_system,  # type: ignore[reportArgumentType]
                    messages=conversation,  # type: ignore[reportArgumentType]
                    inferenceConfig=_inference_config,
                )

        logger.debug(f"Received Bedrock Response: {response}")

        return ConverseResponseToAgentleGenerationAdapter(
            model=_model, response_schema=response_schema
        ).adapt(response)

    @override
    def price_per_million_tokens_input(
        self, model: str, estimate_tokens: int | None = None
    ) -> float:
        """
        Get the price per million tokens for input/prompt tokens.

        Args:
            model: The Amazon Bedrock model identifier (e.g., "anthropic.claude-3-sonnet-20240229-v1:0")
            estimate_tokens: Optional estimate of token count (unused for current pricing models)

        Returns:
            float: The price per million input tokens for the specified model in USD.
                Returns 0.0 for models that use alternative pricing (e.g., per-image pricing).
        """
        # Pricing in USD per million tokens as of June 2025
        model_to_price_per_million: Mapping[str, float] = {
            # Anthropic Claude 4 Series (Contact AWS for pricing - using placeholder)
            "anthropic.claude-opus-4-20250514-v1:0": 0.0,  # Contact AWS
            "anthropic.claude-sonnet-4-20250514-v1:0": 0.0,  # Contact AWS
            "us.anthropic.claude-sonnet-4-20250514-v1:0": 0.0,  # US region variant
            "us.anthropic.claude-opus-4-20250514-v1:0": 0.0,  # US region variant
            # Anthropic Claude 3.7 Series
            "anthropic.claude-3-7-sonnet-20250219-v1:0": 3.00,
            # Anthropic Claude 3.5 Series
            "anthropic.claude-3-5-sonnet-20241022-v2:0": 3.00,
            "anthropic.claude-3-5-haiku-20241022-v1:0": 0.80,
            # Anthropic Claude 3 Series
            "anthropic.claude-3-opus-20240229-v1:0": 15.00,
            "anthropic.claude-3-sonnet-20240229-v1:0": 3.00,
            "anthropic.claude-3-haiku-20240307-v1:0": 0.25,
            # Anthropic Claude Legacy
            "anthropic.claude-2-1-v1:0": 8.00,
            "anthropic.claude-instant-1-2-v1:0": 0.80,
            # Amazon Nova Series
            "amazon.nova-micro-v1:0": 0.035,
            "amazon.nova-lite-v1:0": 0.20,
            "amazon.nova-pro-v1:0": 0.80,
            "amazon.nova-premier-v1:0": 0.0,  # Coming Q1 2025, pricing TBD
            # Amazon Nova Creative (Mixed pricing - text portions only)
            "amazon.nova-sonic-v1:0": 3.40,  # Text input portion
            "amazon.nova-canvas-v1:0": 0.0,  # Per-image pricing
            "amazon.nova-reel-v1:0": 0.0,  # Per-video pricing
            # Amazon Titan Text Models
            "amazon.titan-text-premier-v1:0": 0.50,
            "amazon.titan-text-express-v1:0": 0.80,
            "amazon.titan-text-lite-v1:0": 0.30,
            # Amazon Titan Embedding Models (input-only)
            "amazon.titan-embed-text-v2:0": 0.10,
            "amazon.titan-embed-text-v1": 0.10,
            "amazon.titan-embed-image-v1": 0.10,  # Text portion only
            # Amazon Titan Image Generation
            "amazon.titan-image-generator-v2:0": 0.0,  # Per-image pricing
            "amazon.titan-image-generator-v1": 0.0,  # Per-image pricing
            # AI21 Labs Jamba Series
            "ai21.jamba-1-5-large-v1:0": 2.00,
            "ai21.jamba-1-5-mini-v1:0": 0.20,
            "ai21.jamba-instruct-v1:0": 0.50,
            # AI21 Labs Jurassic Series
            "ai21.j2-ultra-v1": 18.80,
            "ai21.j2-mid-v1": 12.50,
            # Cohere Command Series
            "cohere.command-r-plus-v1:0": 3.00,
            "cohere.command-r-v1:0": 0.50,
            "cohere.command-text-v14": 1.50,
            "cohere.command-light-text-v14": 0.30,
            # Cohere Embedding Models
            "cohere.embed-english-v3": 0.10,
            "cohere.embed-multilingual-v3": 0.10,
            "cohere.rerank-v3-5:0": 0.0,  # Per-query pricing, not per-token
            # Meta Llama 4 Series (Contact AWS)
            "us.meta.llama4-maverick-17b-instruct-v1:0": 0.0,  # Contact AWS
            "us.meta.llama4-scout-17b-instruct-v1:0": 0.0,  # Contact AWS
            # Meta Llama 3 Series
            "meta.llama3-1-405b-instruct-v1:0": 0.0,  # Contact AWS
            "meta.llama3-1-70b-instruct-v1:0": 0.0,  # Contact AWS
            "meta.llama3-1-8b-instruct-v1:0": 0.0,  # Contact AWS
            "meta.llama2-70b-chat-v1": 1.95,
            "meta.llama2-13b-chat-v1": 0.75,
            # Stability AI Models (per-image pricing)
            "stability.stable-image-ultra-v1:0": 0.0,
            "stability.stable-diffusion-3-5-large": 0.0,
            "stability.stable-image-core-v1:0": 0.0,
            "stability.stable-diffusion-xl-v1": 0.0,
            # Mistral AI
            "mistral.mistral-large-2407-v1:0": 8.00,
            "mistral.mixtral-8x7b-instruct-v0:1": 0.45,
            # DeepSeek
            "deepseek.deepseek-r1-distill-qwen-32b-v1:0": 1.35,
        }

        price = model_to_price_per_million.get(model)
        if price is None:
            logger.warning(
                f"Model {model} not found in pricing database. Returning 0.0. "
                + "Please check the model ID or contact AWS for pricing information."
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
            model: The Amazon Bedrock model identifier
            estimate_tokens: Optional estimate of token count (unused for current pricing models)

        Returns:
            float: The price per million output tokens for the specified model in USD.
                Returns 0.0 for models that use alternative pricing or input-only models.
        """
        # Pricing in USD per million tokens as of June 2025
        model_to_price_per_million: Mapping[str, float] = {
            # Anthropic Claude 4 Series (Contact AWS for pricing - using placeholder)
            "anthropic.claude-opus-4-20250514-v1:0": 0.0,  # Contact AWS
            "anthropic.claude-sonnet-4-20250514-v1:0": 0.0,  # Contact AWS
            "us.anthropic.claude-sonnet-4-20250514-v1:0": 0.0,  # US region variant
            "us.anthropic.claude-opus-4-20250514-v1:0": 0.0,  # US region variant
            # Anthropic Claude 3.7 Series
            "anthropic.claude-3-7-sonnet-20250219-v1:0": 15.00,
            # Anthropic Claude 3.5 Series
            "anthropic.claude-3-5-sonnet-20241022-v2:0": 15.00,
            "anthropic.claude-3-5-haiku-20241022-v1:0": 4.00,
            # Anthropic Claude 3 Series
            "anthropic.claude-3-opus-20240229-v1:0": 75.00,
            "anthropic.claude-3-sonnet-20240229-v1:0": 15.00,
            "anthropic.claude-3-haiku-20240307-v1:0": 1.25,
            # Anthropic Claude Legacy
            "anthropic.claude-2-1-v1:0": 24.00,
            "anthropic.claude-instant-1-2-v1:0": 2.40,
            # Amazon Nova Series
            "amazon.nova-micro-v1:0": 0.14,
            "amazon.nova-lite-v1:0": 0.80,
            "amazon.nova-pro-v1:0": 3.20,
            "amazon.nova-premier-v1:0": 0.0,  # Coming Q1 2025, pricing TBD
            # Amazon Nova Creative (Mixed pricing)
            "amazon.nova-sonic-v1:0": 0.06,  # Text output portion
            "amazon.nova-canvas-v1:0": 0.0,  # Per-image pricing
            "amazon.nova-reel-v1:0": 0.0,  # Per-video pricing
            # Amazon Titan Text Models
            "amazon.titan-text-premier-v1:0": 1.50,
            "amazon.titan-text-express-v1:0": 1.60,
            "amazon.titan-text-lite-v1:0": 0.40,
            # Amazon Titan Embedding Models (no output pricing)
            "amazon.titan-embed-text-v2:0": 0.0,
            "amazon.titan-embed-text-v1": 0.0,
            "amazon.titan-embed-image-v1": 0.0,
            # Amazon Titan Image Generation
            "amazon.titan-image-generator-v2:0": 0.0,  # Per-image pricing
            "amazon.titan-image-generator-v1": 0.0,  # Per-image pricing
            # AI21 Labs Jamba Series
            "ai21.jamba-1-5-large-v1:0": 8.00,
            "ai21.jamba-1-5-mini-v1:0": 0.40,
            "ai21.jamba-instruct-v1:0": 0.70,
            # AI21 Labs Jurassic Series
            "ai21.j2-ultra-v1": 18.80,
            "ai21.j2-mid-v1": 12.50,
            # Cohere Command Series
            "cohere.command-r-plus-v1:0": 15.00,
            "cohere.command-r-v1:0": 1.50,
            "cohere.command-text-v14": 2.00,
            "cohere.command-light-text-v14": 0.60,
            # Cohere Embedding Models (no output pricing)
            "cohere.embed-english-v3": 0.0,
            "cohere.embed-multilingual-v3": 0.0,
            "cohere.rerank-v3-5:0": 0.0,  # Per-query pricing
            # Meta Llama 4 Series (Contact AWS)
            "us.meta.llama4-maverick-17b-instruct-v1:0": 0.0,  # Contact AWS
            "us.meta.llama4-scout-17b-instruct-v1:0": 0.0,  # Contact AWS
            # Meta Llama 3 Series
            "meta.llama3-1-405b-instruct-v1:0": 0.0,  # Contact AWS
            "meta.llama3-1-70b-instruct-v1:0": 0.0,  # Contact AWS
            "meta.llama3-1-8b-instruct-v1:0": 0.0,  # Contact AWS
            "meta.llama2-70b-chat-v1": 2.56,
            "meta.llama2-13b-chat-v1": 1.00,
            # Stability AI Models (per-image pricing)
            "stability.stable-image-ultra-v1:0": 0.0,
            "stability.stable-diffusion-3-5-large": 0.0,
            "stability.stable-image-core-v1:0": 0.0,
            "stability.stable-diffusion-xl-v1": 0.0,
            # Mistral AI
            "mistral.mistral-large-2407-v1:0": 24.00,
            "mistral.mixtral-8x7b-instruct-v0:1": 0.70,
            # DeepSeek
            "deepseek.deepseek-r1-distill-qwen-32b-v1:0": 5.40,
        }

        price = model_to_price_per_million.get(model)
        if price is None:
            logger.warning(
                f"Model {model} not found in pricing database. Returning 0.0. "
                + "Please check the model ID or contact AWS for pricing information."
            )
            return 0.0

        return price

    @override
    def map_model_kind_to_provider_model(
        self,
        model_kind: ModelKind,
    ) -> str:
        """
        Map generic model kinds to specific Amazon Bedrock provider model identifiers.

        Args:
            model_kind: The generic model category to map

        Returns:
            str: The specific Bedrock model identifier for the given category.
                Falls back to Claude 3.5 Sonnet for unknown categories.
        """
        mapping: Mapping[ModelKind, str] = {
            # Stable models - optimized for production use
            "category_nano": "amazon.nova-micro-v1:0",  # Ultra-lightweight, cost-optimized
            "category_mini": "amazon.nova-lite-v1:0",  # Lightweight, efficient
            "category_standard": "anthropic.claude-3-5-sonnet-20241022-v2:0",  # General-purpose, balanced
            "category_pro": "amazon.nova-pro-v1:0",  # Professional-grade, multimodal
            "category_flagship": "anthropic.claude-3-opus-20240229-v1:0",  # State-of-the-art performance
            "category_reasoning": "anthropic.claude-3-7-sonnet-20250219-v1:0",  # Extended thinking capabilities
            "category_vision": "amazon.nova-pro-v1:0",  # Multimodal vision capabilities
            "category_coding": "anthropic.claude-3-opus-20240229-v1:0",  # Optimized for code generation
            "category_instruct": "amazon.titan-text-premier-v1:0",  # Agent-optimized instruction following
            "category_nano_experimental": "amazon.nova-micro-v1:0",  # Same as stable for now
            "category_mini_experimental": "amazon.nova-lite-v1:0",  # Latest lightweight model
            "category_standard_experimental": "anthropic.claude-3-7-sonnet-20250219-v1:0",  # Latest reasoning model
            "category_pro_experimental": "amazon.nova-premier-v1:0",  # Coming Q1 2025
            "category_flagship_experimental": "us.anthropic.claude-sonnet-4-20250514-v1:0",  # Claude 4 when available
            "category_reasoning_experimental": "us.anthropic.claude-opus-4-20250514-v1:0",  # Claude Opus 4 when available
            "category_vision_experimental": "amazon.nova-pro-v1:0",  # Latest multimodal capabilities
            "category_coding_experimental": "us.anthropic.claude-opus-4-20250514-v1:0",  # Latest coding capabilities
            "category_instruct_experimental": "amazon.nova-premier-v1:0",  # When available
        }

        model_id = mapping.get(model_kind)
        if model_id is None:
            logger.warning(
                f"Model kind {model_kind} not found in mapping. "
                + "Falling back to Claude 3.5 Sonnet as default."
            )
            return "anthropic.claude-3-5-sonnet-20241022-v2:0"

        return model_id
