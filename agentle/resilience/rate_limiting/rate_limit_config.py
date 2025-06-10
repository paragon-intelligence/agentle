from typing import NotRequired, TypedDict


class RateLimitConfig(TypedDict):
    """Configuration for rate limiting."""

    max_requests_per_minute: NotRequired[int]
    max_requests_per_hour: NotRequired[int]
    max_requests_per_day: NotRequired[int]
