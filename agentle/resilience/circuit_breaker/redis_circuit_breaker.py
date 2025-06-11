from __future__ import annotations

import time
from typing import TYPE_CHECKING, override

from agentle.resilience.circuit_breaker.circuit_breaker_protocol import (
    CircuitBreakerProtocol,
)

if TYPE_CHECKING:
    import redis.asyncio as aioredis


class RedisCircuitBreaker(CircuitBreakerProtocol):
    """
    Redis-based circuit breaker implementation for distributed systems.

    This implementation stores circuit state in Redis, making it suitable
    for distributed systems with multiple processes/workers.
    """

    redis_client: aioredis.Redis
    failure_threshold: int = 5
    recovery_timeout: float = 300.0  # 5 minutes
    key_prefix: str = "circuit_breaker"
    key_ttl: int = 86400  # 24 hours

    def __init__(
        self,
        redis_client: aioredis.Redis,
        failure_threshold: int = 5,
        recovery_timeout: float = 300.0,
        key_prefix: str = "circuit_breaker",
        key_ttl: int = 86400,
    ) -> None:
        self.redis_client = redis_client
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.key_prefix = key_prefix
        self.key_ttl = key_ttl

    def _get_keys(self, circuit_id: str) -> tuple[str, str, str]:
        """Get Redis keys for circuit state."""
        base_key = f"{self.key_prefix}:{circuit_id}"
        return (
            f"{base_key}:failures",
            f"{base_key}:last_failure",
            f"{base_key}:is_open",
        )

    @override
    async def is_open(self, circuit_id: str) -> bool:
        """Check if the circuit is open (blocking operations)."""
        _, last_failure_key, is_open_key = self._get_keys(circuit_id)

        is_open = await self.redis_client.get(is_open_key)
        if not is_open or is_open.decode() != "1":
            return False

        # Check if recovery timeout has passed
        last_failure_time = await self.redis_client.get(last_failure_key)
        if last_failure_time:
            last_failure_float = float(last_failure_time.decode())
            if time.time() - last_failure_float > self.recovery_timeout:
                # Reset circuit to half-open state
                await self.reset_circuit(circuit_id)
                return False

        return True

    @override
    async def record_success(self, circuit_id: str) -> None:
        """Record a successful operation."""
        await self.reset_circuit(circuit_id)

    @override
    async def record_failure(self, circuit_id: str) -> None:
        """Record a failed operation."""
        failure_key, last_failure_key, is_open_key = self._get_keys(circuit_id)

        pipe = self.redis_client.pipeline()
        pipe.incr(failure_key)
        pipe.set(last_failure_key, str(time.time()), ex=self.key_ttl)
        results = await pipe.execute()

        failure_count = results[0]

        if failure_count >= self.failure_threshold:
            await self.redis_client.set(is_open_key, "1", ex=self.key_ttl)

    @override
    async def get_failure_count(self, circuit_id: str) -> int:
        """Get the current failure count for the circuit."""
        failure_key, _, _ = self._get_keys(circuit_id)
        count = await self.redis_client.get(failure_key)
        return int(count.decode()) if count else 0

    @override
    async def reset_circuit(self, circuit_id: str) -> None:
        """Manually reset the circuit to closed state."""
        failure_key, last_failure_key, is_open_key = self._get_keys(circuit_id)

        pipe = self.redis_client.pipeline()
        pipe.delete(failure_key)
        pipe.delete(last_failure_key)
        pipe.delete(is_open_key)
        await pipe.execute()
