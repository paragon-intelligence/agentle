"""
Provides the default implementation for calculating language model prices.

This module implements a standard pricing calculation strategy that determines
the cost of language model generations based on input and output token counts
and the provider's pricing information.
"""

from typing import override

from agentle.generations.pricing.price_retrievable import PriceRetrievable
from agentle.generations.pricing.pricing_calculator import (
    PricingCalculationStrategy,
)


class DefaultInternalLLMPriceCalculationStrategy(PricingCalculationStrategy):
    """
    Default implementation of the pricing calculation strategy.

    This strategy calculates the price of language model generations using
    the standard pricing formula:
    (input_tokens / 1M × input_price_per_million) + (output_tokens / 1M × output_price_per_million)

    It retrieves the pricing information from the provider and applies this formula
    to determine the total cost. This is the most common pricing model used by
    language model providers.

    Example:
        ```python
        calculator = DefaultInternalLLMPriceCalculationStrategy()
        provider = SomeProvider()  # A class implementing PriceRetrievable

        price = calculator.calculate_price(
            input_tokens=1000,
            completion_tokens=500,
            model="gpt-4",
            provider=provider
        )
        ```
    """

    @override
    def calculate_price(
        self,
        input_tokens: int,
        completion_tokens: int,
        model: str,
        provider: PriceRetrievable,
    ) -> float:
        """
        Calculate the price for a language model generation.

        This implementation retrieves the pricing information from the provider
        and calculates the total cost based on the number of input and output tokens.

        Args:
            input_tokens: The number of input/prompt tokens used.
            completion_tokens: The number of output/completion tokens generated.
            model: The name or identifier of the language model used.
            provider: An object implementing the PriceRetrievable interface,
                which provides access to the pricing information.

        Returns:
            float: The calculated price in USD.
        """
        selected_model_input_pricing = provider.price_per_million_tokens_input(
            model=model
        )
        selected_model_output_pricing = provider.price_per_million_tokens_output(
            model=model
        )

        input_pricing = input_tokens / 1_000_000 * selected_model_input_pricing
        output_pricing = completion_tokens / 1_000_000 * selected_model_output_pricing

        return input_pricing + output_pricing
