from typing import override

from agentle.generations.pricing.price_retrievable import PriceRetrievable
from agentle.generations.pricing.pricing_calculator import (
    PricingCalculationStrategy,
)


class DefaultInternalLLMPriceCalculationStrategy(PricingCalculationStrategy):
    @override
    def calculate_price(
        self,
        input_tokens: int,
        completion_tokens: int,
        model: str,
        provider: PriceRetrievable,
    ) -> float:
        selected_model_input_pricing = provider.price_per_million_tokens_input(
            model=model
        )
        selected_model_output_pricing = provider.price_per_million_tokens_output(
            model=model
        )

        input_pricing = input_tokens / 1_000_000 * selected_model_input_pricing
        output_pricing = completion_tokens / 1_000_000 * selected_model_output_pricing

        return input_pricing + output_pricing
