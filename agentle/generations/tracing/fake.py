"""
Fake implementation of the StatefulObservabilityClient for testing and development.

This module provides the FakeObservabilityClient class, which implements the
StatefulObservabilityClient interface but doesn't actually send data to any
observability platform. Instead, it logs operations locally or stores them
in memory for inspection.

This is useful for:
- Testing applications without requiring real observability infrastructure
- Development environments where observability isn't needed
- Unit tests that need to verify observability calls without external dependencies
- Performance testing where observability overhead should be minimized

The fake client maintains the same interface as real implementations, so it can
be used as a drop-in replacement without changing application code.

Example:
```python
from agentle.generations.tracing.fake import FakeObservabilityClient

# Create a fake client for testing
client = FakeObservabilityClient()

# Use it exactly like a real client
trace = await client.trace(
    name="test_operation",
    user_id="test_user",
    metadata={"test": True}
)

generation = await trace.generation(
    name="test_generation",
    input={"prompt": "Hello world"},
    metadata={"model": "test-model"}
)

await generation.end(
    output={"response": "Hello back!"},
    metadata={"tokens": 10}
)

await trace.end()

# Inspect what was tracked
print(client.get_traces())  # See all traces that were created
print(client.get_operations())  # See all operations in order
```
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any, Optional, override

from agentle.generations.tracing.contracts.stateful_observability_client import (
    StatefulObservabilityClient,
)


class FakeObservabilityClient(StatefulObservabilityClient):
    """
    Fake implementation of StatefulObservabilityClient for testing and development.

    This client implements the full StatefulObservabilityClient interface but doesn't
    send data to any external observability platform. Instead, it stores operations
    in memory and optionally logs them for debugging purposes.

    The fake client maintains the hierarchical structure and method chaining behavior
    of real implementations, making it a perfect drop-in replacement for testing.

    Features:
    - Full interface compatibility with real observability clients
    - In-memory storage of all operations for inspection
    - Optional logging of operations for debugging
    - Maintains parent-child relationships between traces, spans, and generations
    - Supports all metadata, timing, and scoring operations

    Attributes:
        _operations: List of all operations performed, in chronological order
        _traces: Dictionary mapping trace IDs to trace data
        _current_trace_id: The trace ID for this client instance
        _parent_id: Optional parent operation ID for hierarchical structure
        _operation_id: Unique ID for this specific operation
        _logger: Logger instance for this class
        _enable_logging: Whether to log operations to the logger

    Example:
        ```python
        # Basic usage
        client = FakeObservabilityClient(enable_logging=True)

        # Create and track operations
        trace = await client.trace(name="test_trace")
        span = await trace.span(name="test_span")
        await span.end(output={"result": "success"})
        await trace.end()

        # Inspect what happened
        operations = client.get_operations()
        traces = client.get_traces()

        # Check if specific operations occurred
        assert any(op["type"] == "trace" and op["name"] == "test_trace" for op in operations)
        ```
    """

    def __init__(
        self,
        *,
        enable_logging: bool = False,
        trace_id: Optional[str] = None,
        parent_id: Optional[str] = None,
        operation_id: Optional[str] = None,
        shared_state: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Initialize a new FakeObservabilityClient.

        Args:
            enable_logging: Whether to log operations to the Python logger.
                Useful for debugging but can be noisy in tests.
            trace_id: Optional trace ID to use. If not provided, a new UUID is generated.
            parent_id: Optional parent operation ID for hierarchical tracking.
            operation_id: Optional operation ID for this specific client instance.
            shared_state: Optional shared state dictionary for storing operations across
                multiple client instances. If not provided, a new dictionary is created.
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._enable_logging = enable_logging
        self._current_trace_id = trace_id or str(uuid.uuid4())
        self._parent_id = parent_id
        self._operation_id = operation_id or str(uuid.uuid4())

        # Shared state allows multiple client instances to share the same operation log
        if shared_state is None:
            self._shared_state: dict[str, Any] = {
                "operations": [],
                "traces": {},
            }
        else:
            self._shared_state = shared_state

    def _log_operation(
        self,
        operation_type: str,
        name: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Log an operation to the internal storage and optionally to the logger.

        Args:
            operation_type: Type of operation (trace, span, generation, event, end)
            name: Name of the operation
            **kwargs: Additional operation data

        Returns:
            The operation ID for the logged operation
        """
        operation_id = str(uuid.uuid4())
        timestamp = datetime.now()

        operation_data = {
            "id": operation_id,
            "type": operation_type,
            "name": name,
            "trace_id": self._current_trace_id,
            "parent_id": self._parent_id,
            "timestamp": timestamp,
            **kwargs,
        }

        # Store in shared state
        self._shared_state["operations"].append(operation_data)

        # Update trace data if this is a trace operation
        if operation_type == "trace":
            self._shared_state["traces"][self._current_trace_id] = operation_data

        # Log if enabled
        if self._enable_logging:
            self._logger.info(
                f"Fake observability: {operation_type} '{name}' "
                + f"(trace: {self._current_trace_id}, parent: {self._parent_id})"
            )

        return operation_id

    @override
    async def trace(
        self,
        *,
        name: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        input: object | None = None,
        output: object | None = None,
        metadata: dict[str, object] | None = None,
        tags: Sequence[str] | None = None,
        timestamp: datetime | None = None,
    ) -> StatefulObservabilityClient:
        """Create a fake trace."""
        operation_id = self._log_operation(
            "trace",
            name=name,
            user_id=user_id,
            session_id=session_id,
            input=input,
            output=output,
            metadata=metadata,
            tags=list(tags) if tags else None,
            timestamp=timestamp,
        )

        # Create a new client for the trace with a new trace ID
        new_trace_id = str(uuid.uuid4())
        return FakeObservabilityClient(
            enable_logging=self._enable_logging,
            trace_id=new_trace_id,
            parent_id=self._operation_id,
            operation_id=operation_id,
            shared_state=self._shared_state,
        )

    @override
    async def generation(
        self,
        *,
        name: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        input: object | None = None,
        output: object | None = None,
        metadata: dict[str, object] | None = None,
        tags: Sequence[str] | None = None,
        timestamp: datetime | None = None,
    ) -> StatefulObservabilityClient:
        """Create a fake generation."""
        operation_id = self._log_operation(
            "generation",
            name=name,
            user_id=user_id,
            session_id=session_id,
            input=input,
            output=output,
            metadata=metadata,
            tags=list(tags) if tags else None,
            timestamp=timestamp,
        )

        return FakeObservabilityClient(
            enable_logging=self._enable_logging,
            trace_id=self._current_trace_id,
            parent_id=self._operation_id,
            operation_id=operation_id,
            shared_state=self._shared_state,
        )

    @override
    async def span(
        self,
        *,
        name: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        input: object | None = None,
        output: object | None = None,
        metadata: dict[str, object] | None = None,
        tags: Sequence[str] | None = None,
        timestamp: datetime | None = None,
    ) -> StatefulObservabilityClient:
        """Create a fake span."""
        operation_id = self._log_operation(
            "span",
            name=name,
            user_id=user_id,
            session_id=session_id,
            input=input,
            output=output,
            metadata=metadata,
            tags=list(tags) if tags else None,
            timestamp=timestamp,
        )

        return FakeObservabilityClient(
            enable_logging=self._enable_logging,
            trace_id=self._current_trace_id,
            parent_id=self._operation_id,
            operation_id=operation_id,
            shared_state=self._shared_state,
        )

    @override
    async def event(
        self,
        *,
        name: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        input: object | None = None,
        output: object | None = None,
        metadata: dict[str, object] | None = None,
        tags: Sequence[str] | None = None,
        timestamp: datetime | None = None,
    ) -> StatefulObservabilityClient:
        """Create a fake event."""
        operation_id = self._log_operation(
            "event",
            name=name,
            user_id=user_id,
            session_id=session_id,
            input=input,
            output=output,
            metadata=metadata,
            tags=list(tags) if tags else None,
            timestamp=timestamp,
        )

        return FakeObservabilityClient(
            enable_logging=self._enable_logging,
            trace_id=self._current_trace_id,
            parent_id=self._operation_id,
            operation_id=operation_id,
            shared_state=self._shared_state,
        )

    @override
    async def end(
        self,
        *,
        name: str | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
        input: object | None = None,
        output: object | None = None,
        metadata: Mapping[str, object] | None = None,
        tags: Sequence[str] | None = None,
        timestamp: datetime | None = None,
        usage_details: Mapping[str, Any] | None = None,
        cost_details: Mapping[str, Any] | None = None,
    ) -> StatefulObservabilityClient:
        """End the current fake operation."""
        self._log_operation(
            "end",
            name=name,
            user_id=user_id,
            session_id=session_id,
            input=input,
            output=output,
            metadata=dict(metadata) if metadata else None,
            tags=list(tags) if tags else None,
            timestamp=timestamp,
            usage_details=dict(usage_details) if usage_details else None,
            cost_details=dict(cost_details) if cost_details else None,
        )

        return self

    @override
    async def flush(self) -> None:
        """Fake flush operation - just logs that flush was called."""
        if self._enable_logging:
            self._logger.info("Fake observability: flush() called")

    @override
    async def score_trace(
        self,
        *,
        name: str,
        value: float | str,
        trace_id: str | None = None,
        comment: str | None = None,
    ) -> None:
        """Add a fake score to a trace."""
        target_trace_id = trace_id or self._current_trace_id

        self._log_operation(
            "score",
            name=name,
            value=value,
            target_trace_id=target_trace_id,
            comment=comment,
        )

    # Additional methods for inspecting the fake client's state

    def get_operations(self) -> list[dict[str, Any]]:
        """
        Get all operations that have been logged by this client.

        Returns:
            List of operation dictionaries in chronological order.
        """
        return self._shared_state["operations"].copy()

    def get_traces(self) -> dict[str, dict[str, Any]]:
        """
        Get all traces that have been created by this client.

        Returns:
            Dictionary mapping trace IDs to trace data.
        """
        return self._shared_state["traces"].copy()

    def get_operations_by_type(self, operation_type: str) -> list[dict[str, Any]]:
        """
        Get all operations of a specific type.

        Args:
            operation_type: The type of operation to filter by (trace, span, generation, event, end, score)

        Returns:
            List of operations of the specified type.
        """
        return [
            op
            for op in self._shared_state["operations"]
            if op["type"] == operation_type
        ]

    def get_operations_by_trace(self, trace_id: str) -> list[dict[str, Any]]:
        """
        Get all operations for a specific trace.

        Args:
            trace_id: The trace ID to filter by

        Returns:
            List of operations for the specified trace.
        """
        return [
            op for op in self._shared_state["operations"] if op["trace_id"] == trace_id
        ]

    def clear(self) -> None:
        """
        Clear all stored operations and traces.

        Useful for resetting state between tests.
        """
        self._shared_state["operations"].clear()
        self._shared_state["traces"].clear()

    def operation_count(self) -> int:
        """
        Get the total number of operations logged.

        Returns:
            The number of operations that have been logged.
        """
        return len(self._shared_state["operations"])

    def has_operation(self, operation_type: str, name: str | None = None) -> bool:
        """
        Check if an operation of a specific type and optionally name exists.

        Args:
            operation_type: The type of operation to look for
            name: Optional name to match

        Returns:
            True if such an operation exists, False otherwise.
        """
        for op in self._shared_state["operations"]:
            if op["type"] == operation_type:
                if name is None or op["name"] == name:
                    return True
        return False
