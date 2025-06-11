from dataclasses import dataclass


@dataclass
class CircuitState:
    """State information for a single circuit."""

    failure_count: int = 0
    last_failure_time: float = 0.0
    is_open: bool = False
