# Adapter for OpenRouter response to generation
"""
Adapter for converting OpenRouter API responses to Agentle Generation objects.

This module handles the transformation of OpenRouter's response format into
Agentle's standardized Generation format, including choices, usage statistics,
and metadata.
"""

from __future__ import annotations

import datetime
import logging
import uuid
from typing import TYPE_CHECKING, override

from rsb.adapters.adapter import Adapter

from agentle.generations.models.generation.choice import Choice
from agentle.generations.models.generation.generation import Generation
from agentle.generations.models.generation.pricing import Pricing
from agentle.generations.models.generation.usage import Usage
from agentle.generations.providers.openrouter._adapters.openrouter_message_to_generated_assistant_message_adapter import (
    OpenRouterMessageToGeneratedAssistantMessageAdapter,
)
from agentle.generations.providers.openrouter._types import OpenRouterResponse

if TYPE_CHECKING:
    from agentle.generations.providers.openrouter.openrouter_generation_provider import (
        OpenRouterGenerationProvider,
    )

logger = logging.getLogger(__name__)


class OpenRouterResponseToGenerationAdapter[T](
    Adapter[OpenRouterResponse, Generation[T]]
):
    """
    Adapter for converting OpenRouter responses to Agentle Generation objects.

    Processes the complete response including choices, usage statistics, and
    any structured output data.

    Attributes:
        response_schema: Optional Pydantic model class for parsing structured data.
        preferred_id: Optional UUID to use for the Generation object.
        message_adapter: Adapter for converting response messages.
        provider: Optional provider instance for pricing calculation.
        model: Optional model identifier for pricing calculation.
    """

    response_schema: type[T] | None
    preferred_id: uuid.UUID | None
    message_adapter: OpenRouterMessageToGeneratedAssistantMessageAdapter[T]
    provider: OpenRouterGenerationProvider | None
    model: str | None

    def __init__(
        self,
        *,
        response_schema: type[T] | None = None,
        preferred_id: uuid.UUID | None = None,
        message_adapter: OpenRouterMessageToGeneratedAssistantMessageAdapter[T]
        | None = None,
        provider: OpenRouterGenerationProvider | None = None,
        model: str | None = None,
    ):
        """
        Initialize the adapter.

        Args:
            response_schema: Optional Pydantic model class for structured output.
            preferred_id: Optional UUID to use for the Generation.
            message_adapter: Optional message adapter (created if not provided).
            provider: Optional provider instance for pricing calculation.
            model: Optional model identifier for pricing calculation.
        """
        self.response_schema = response_schema
        self.preferred_id = preferred_id
        self.message_adapter = (
            message_adapter
            or OpenRouterMessageToGeneratedAssistantMessageAdapter(
                response_schema=response_schema
            )
        )
        self.provider = provider
        self.model = model

    @override
    def adapt(self, _f: OpenRouterResponse) -> Generation[T]:
        """
        Convert an OpenRouter response to an Agentle Generation.

        Args:
            _f: The OpenRouter API response to convert.

        Returns:
            Generation object with normalized data.
        """
        openrouter_response = _f

        logger.debug("OpenRouter response: ")
        logger.debug(openrouter_response)

        # Convert choices
        choices: list[Choice[T]] = [
            Choice(
                index=choice["index"],
                message=self.message_adapter.adapt(choice["message"]),
            )
            for choice in openrouter_response["choices"]
        ]

        # Extract usage information
        usage_data = openrouter_response["usage"]
        usage = Usage(
            prompt_tokens=usage_data["prompt_tokens"],
            completion_tokens=usage_data["completion_tokens"],
        )

        # Build Generation object (pricing will be calculated by the provider)
        return Generation(
            id=self.preferred_id or uuid.uuid4(),
            choices=choices,
            object="chat.generation",
            created=datetime.datetime.fromtimestamp(openrouter_response["created"]),
            model=openrouter_response["model"],
            usage=usage,
        )

    async def adapt_async(self, _f: OpenRouterResponse) -> Generation[T]:
        """
        Convert an OpenRouter response to an Agentle Generation asynchronously.

        This async version calculates pricing information if provider and model are available.

        Args:
            _f: The OpenRouter API response to convert.

        Returns:
            Generation object with normalized data and pricing information.
        """
        openrouter_response = _f

        # Convert choices
        choices: list[Choice[T]] = [
            Choice(
                index=choice["index"],
                message=self.message_adapter.adapt(choice["message"]),
            )
            for choice in openrouter_response["choices"]
        ]

        # Extract usage information
        usage_data = openrouter_response["usage"]
        usage = Usage(
            prompt_tokens=usage_data["prompt_tokens"],
            completion_tokens=usage_data["completion_tokens"],
        )

        # Calculate pricing if provider and model are available
        pricing = Pricing()
        if self.provider is not None and self.model is not None:
            provider = self.provider
            model = self.model
            try:
                input_tokens = usage.prompt_tokens
                output_tokens = usage.completion_tokens

                if input_tokens > 0 or output_tokens > 0:
                    input_price_per_million = (
                        await provider.price_per_million_tokens_input(
                            model, input_tokens
                        )
                    )
                    output_price_per_million = (
                        await provider.price_per_million_tokens_output(
                            model, output_tokens
                        )
                    )

                    input_cost = input_price_per_million * (input_tokens / 1_000_000)
                    output_cost = output_price_per_million * (output_tokens / 1_000_000)
                    total_cost = input_cost + output_cost

                    pricing = Pricing(
                        input_pricing=round(input_cost, 8),
                        output_pricing=round(output_cost, 8),
                        total_pricing=round(total_cost, 8),
                    )

            except Exception as e:
                logger.warning(f"Failed to calculate pricing: {e}")
                pricing = Pricing()

        # Build Generation object
        return Generation(
            id=self.preferred_id or uuid.uuid4(),
            choices=choices,
            object="chat.generation",
            created=datetime.datetime.fromtimestamp(openrouter_response["created"]),
            model=openrouter_response["model"],
            usage=usage,
            pricing=pricing,
        )
