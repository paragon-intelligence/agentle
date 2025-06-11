from collections.abc import Mapping
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import override

from agentle.resilience.circuit_breaker.circuit_breaker_protocol import (
    CircuitBreakerProtocol,
)
from agentle.resilience.circuit_breaker.circuit_state import CircuitState


@dataclass
class InMemoryCircuitBreaker(CircuitBreakerProtocol):
    """
    In-memory circuit breaker implementation.

    WARNING: This implementation stores state in memory and is NOT suitable
    for distributed systems with multiple processes/workers. Use RedisCircuitBreaker
    or DatabaseCircuitBreaker for production distributed scenarios.
    """

    failure_threshold: int = 5
    recovery_timeout: float = 300.0  # 5 minutes
    _circuits: Mapping[str, CircuitState] = field(
        default_factory=lambda: defaultdict(CircuitState)
    )

    @override
    async def is_open(self, circuit_id: str) -> bool:
        """Check if the circuit is open (blocking operations)."""
        circuit = self._circuits[circuit_id]

        if not circuit.is_open:
            return False

        # Check if recovery timeout has passed
        if time.time() - circuit.last_failure_time > self.recovery_timeout:
            # Reset circuit to half-open state (allow one test)
            circuit.is_open = False
            circuit.failure_count = 0
            return False

        return True

    @override
    async def record_success(self, circuit_id: str) -> None:
        """Record a successful operation."""
        circuit = self._circuits[circuit_id]
        circuit.failure_count = 0
        circuit.is_open = False
        circuit.last_failure_time = 0.0

    @override
    async def record_failure(self, circuit_id: str) -> None:
        """Record a failed operation."""
        circuit = self._circuits[circuit_id]
        circuit.failure_count += 1
        circuit.last_failure_time = time.time()

        if circuit.failure_count >= self.failure_threshold:
            circuit.is_open = True

    @override
    async def get_failure_count(self, circuit_id: str) -> int:
        """Get the current failure count for the circuit."""
        return self._circuits[circuit_id].failure_count

    @override
    async def reset_circuit(self, circuit_id: str) -> None:
        """Manually reset the circuit to closed state."""
        circuit = self._circuits[circuit_id]
        circuit.failure_count = 0
        circuit.is_open = False
        circuit.last_failure_time = 0.0
