import abc
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from agentle.resilience.rate_limiting.rate_limit_config import RateLimitConfig


@runtime_checkable
class RateLimiterProtocol(Protocol):
    """
    Abstract protocol for rate limiter implementations.

    Rate limiters control the rate of operations to prevent overwhelming
    downstream services or exceeding quotas.
    """

    @abc.abstractmethod
    async def can_proceed(
        self, identifier: str, config: RateLimitConfig | None = None
    ) -> bool:
        """Check if the operation can proceed within rate limits."""
        ...

    @abc.abstractmethod
    async def record_request(
        self, identifier: str, config: RateLimitConfig | None = None
    ) -> None:
        """Record that a request was made."""
        ...

    @abc.abstractmethod
    async def get_current_usage(self, identifier: str) -> Mapping[str, int]:
        """Get current usage statistics."""
        ...

    @abc.abstractmethod
    async def reset_limits(self, identifier: str) -> None:
        """Reset rate limits for an identifier."""
        ...
