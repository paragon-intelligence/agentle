import abc

from agentle.generations.pricing.price_retrievable import PriceRetrievable


class PricingCalculationStrategy(abc.ABC):
    @abc.abstractmethod
    def calculate_price(
        self,
        input_tokens: int,
        completion_tokens: int,
        model: str,
        provider: PriceRetrievable,
    ) -> float: ...
