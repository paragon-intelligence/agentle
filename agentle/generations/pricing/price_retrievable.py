import abc


class PriceRetrievable(abc.ABC):
    @abc.abstractmethod
    def price_per_million_tokens_input(
        self, model: str, estimate_tokens: int | None = None
    ) -> float: ...

    @abc.abstractmethod
    def price_per_million_tokens_output(
        self, model: str, estimate_tokens: int | None = None
    ) -> float: ...
