"""
Defines the interface for pricing calculation strategies.

This module provides the PricingCalculationStrategy abstract base class, which
defines the interface for different strategies that can be used to calculate
the cost of language model generations based on token usage.
"""

import abc

from agentle.generations.pricing.price_retrievable import PriceRetrievable


class PricingCalculationStrategy(abc.ABC):
    """
    Abstract interface for calculating pricing based on token usage.

    This class defines the strategy pattern for different pricing calculation
    algorithms. Implementations of this interface can provide different ways
    to calculate the cost of language model generations based on input and
    output token counts, the model used, and provider-specific pricing.

    Different implementations might account for tiered pricing, discounts,
    special billing arrangements, or other provider-specific pricing details.
    """

    @abc.abstractmethod
    def calculate_price(
        self,
        input_tokens: int,
        completion_tokens: int,
        model: str,
        provider: PriceRetrievable,
    ) -> float:
        """
        Calculate the price for a language model generation.

        Args:
            input_tokens: The number of input/prompt tokens used.
            completion_tokens: The number of output/completion tokens generated.
            model: The name or identifier of the language model used.
            provider: An object implementing the PriceRetrievable interface,
                which provides access to the pricing information specific to the provider.

        Returns:
            float: The calculated price in USD.
        """
        ...
