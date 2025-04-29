"""
Defines the interface for retrieving pricing information for language models.

This module provides the PriceRetrievable abstract base class, which defines the interface
that provider implementations must implement to provide pricing information for their models.
"""

import abc


class PriceRetrievable(abc.ABC):
    """
    Abstract interface for retrieving pricing information for language models.

    This interface must be implemented by any provider that needs to calculate
    costs for language model usage. It defines methods to retrieve the cost per
    million tokens for both input (prompt) and output (completion) tokens.

    Implementations typically access provider-specific pricing data and return
    the appropriate values based on the model and potentially the volume of usage.
    """

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
