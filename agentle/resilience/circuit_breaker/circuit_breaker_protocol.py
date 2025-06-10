import abc
from typing import Protocol, runtime_checkable


@runtime_checkable
class CircuitBreakerProtocol(Protocol):
    """
    Abstract protocol for circuit breaker implementations.

    Circuit breakers prevent cascading failures by temporarily disabling
    operations that are likely to fail, allowing the system to recover.
    """

    @abc.abstractmethod
    async def is_open(self, circuit_id: str) -> bool:
        """Check if the circuit is open (blocking operations)."""
        ...

    @abc.abstractmethod
    async def record_success(self, circuit_id: str) -> None:
        """Record a successful operation."""
        ...

    @abc.abstractmethod
    async def record_failure(self, circuit_id: str) -> None:
        """Record a failed operation."""
        ...

    @abc.abstractmethod
    async def get_failure_count(self, circuit_id: str) -> int:
        """Get the current failure count for the circuit."""
        ...

    @abc.abstractmethod
    async def reset_circuit(self, circuit_id: str) -> None:
        """Manually reset the circuit to closed state."""
        ...
