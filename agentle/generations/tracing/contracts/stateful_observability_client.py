"""
Abstract base class for stateful observability clients in the Agentle framework.

This module defines the StatefulObservabilityClient abstract base class, which provides
a contract for stateful observability clients that can track traces, generations, spans,
and events in AI applications.

The stateful design enables method chaining and the creation of hierarchical tracing structures,
where traces can contain spans, which can contain generations, and so on. This approach
allows for more detailed and structured observability data.

Implementations of this interface (such as LangfuseObservabilityClient) connect to
specific observability platforms while maintaining a consistent API for the Agentle framework.

Example:
```python
# Example implementation usage (with a hypothetical implementation)
client = ConcreteObservabilityClient()

# Create a trace for a user request
trace_client = await client.trace(
    name="user_query",
    user_id="user123",
    input={"query": "Tell me about Tokyo"}
)

# Within that trace, track a model generation
generation_client = await trace_client.generation(
    name="answer_generation",
    metadata={"model": "gemini-1.5-pro"}
)

# Complete the generation with its output
await generation_client.end(output={"text": "Tokyo is the capital of Japan..."})

# Complete the trace
await trace_client.end()
```
"""

from __future__ import annotations

import abc
from collections.abc import Mapping
from datetime import datetime
from typing import Any, Sequence


class StatefulObservabilityClient(abc.ABC):
    """
    Abstract base class for stateful observability clients.

    This class defines a contract for observability clients that track AI system
    operations through traces, generations, spans, and events. The stateful design
    enables method chaining to create hierarchical structures of traced operations.

    Different implementations of this class connect to specific observability
    platforms (e.g., Langfuse, OpenTelemetry) while maintaining a consistent API
    for the Agentle framework.

    All methods return a new StatefulObservabilityClient instance that represents
    the created entity (trace, span, etc.) and can be used for further method calls.
    This enables both hierarchical structuring and method chaining.

    Example:
        ```python
        # With a concrete implementation
        client = ConcreteObservabilityClient()

        # Create and track a full interaction with nested operations
        trace_client = await client.trace(name="process_query", user_id="user123")
        span_client = await trace_client.span(name="retrieve_information")
        generation_client = await span_client.generation(name="generate_response")
        await generation_client.end(output={"text": "Generated response"})
        await span_client.end()  # End span
        await trace_client.end(output={"final_response": "Processed result"})  # End trace
        ```
    """

    @abc.abstractmethod
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
        """
        Create a new trace to track an end-to-end operation.

        A trace represents a complete operation or user interaction, potentially
        containing multiple spans, generations, and events. It's the top-level
        observability entity.

        Args:
            name: Identifier for the trace. Should be descriptive of the operation.
            user_id: Optional identifier for the user who initiated this operation.
            session_id: Optional identifier to group related traces into a session.
            input: Optional input data that initiated this trace.
            output: Optional output data produced by this trace (typically set with end()).
            metadata: Optional additional structured information about this trace.
            tags: Optional tags for categorizing and filtering traces.
            timestamp: Optional timestamp for when this trace started (defaults to now).

        Returns:
            StatefulObservabilityClient: A new stateful client for the created trace.

        Example:
            ```python
            # Create a trace for a user query
            trace = await client.trace(
                name="process_user_query",
                user_id="user123",
                input={"query": "What's the weather in Tokyo?"},
                metadata={"source": "mobile_app"}
            )
            ```
        """
        ...

    @abc.abstractmethod
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
        """
        Create a new generation to track an AI model generation.

        A generation represents a specific AI model invocation that produces
        content based on input. It typically includes information about the model,
        prompt, settings, and resulting output.

        Args:
            name: Identifier for the generation. Should be descriptive of what is being generated.
            user_id: Optional identifier for the user who initiated this generation.
            session_id: Optional identifier to group related generations into a session.
            input: Optional input data/prompt for this generation.
            output: Optional output data produced by this generation (typically set with end()).
            metadata: Optional additional structured information about this generation.
            tags: Optional tags for categorizing and filtering generations.
            timestamp: Optional timestamp for when this generation started (defaults to now).

        Returns:
            StatefulObservabilityClient: A new stateful client for the created generation.

        Example:
            ```python
            # Create a generation for producing a weather forecast
            generation = await trace.generation(
                name="weather_forecast",
                input={"location": "Tokyo", "units": "celsius"},
                metadata={"model": "weather-model-v2", "temperature": 0.7}
            )
            ```
        """
        ...

    @abc.abstractmethod
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
        """
        Create a new span to track a subtask or phase within a larger operation.

        A span represents a discrete subtask or phase within a larger trace. It's useful
        for breaking down complex operations into smaller, measurable units that can
        be analyzed independently.

        Args:
            name: Identifier for the span. Should be descriptive of the subtask.
            user_id: Optional identifier for the user related to this span.
            session_id: Optional identifier to group related spans into a session.
            input: Optional input data for this span.
            output: Optional output data produced by this span (typically set with end()).
            metadata: Optional additional structured information about this span.
            tags: Optional tags for categorizing and filtering spans.
            timestamp: Optional timestamp for when this span started (defaults to now).

        Returns:
            StatefulObservabilityClient: A new stateful client for the created span.

        Example:
            ```python
            # Create a span for data retrieval
            span = await trace.span(
                name="retrieve_weather_data",
                input={"location": "Tokyo"},
                metadata={"data_source": "weather_api"}
            )
            ```
        """
        ...

    @abc.abstractmethod
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
        """
        Create a new event to mark a specific point of interest.

        An event represents a discrete moment or occurrence within a trace that's
        worth noting but doesn't have a duration. Events are useful for marking
        significant points in time, such as when important decisions are made.

        Args:
            name: Identifier for the event. Should be descriptive of what occurred.
            user_id: Optional identifier for the user related to this event.
            session_id: Optional identifier to group related events into a session.
            input: Optional input data related to this event.
            output: Optional output data related to this event.
            metadata: Optional additional structured information about this event.
            tags: Optional tags for categorizing and filtering events.
            timestamp: Optional timestamp for when this event occurred (defaults to now).

        Returns:
            StatefulObservabilityClient: A new stateful client for the created event.

        Example:
            ```python
            # Create an event for a specific occurrence
            event = await span.event(
                name="api_rate_limit_reached",
                metadata={"limit": 100, "remaining": 0, "reset_in": "60s"}
            )
            ```
        """
        ...

    @abc.abstractmethod
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
        """
        End the current trace, span, or generation with optional final data.

        This method marks the current observability entity as complete and optionally
        adds final data such as output, metadata, or tags. The specific behavior
        depends on the type of entity being ended.

        For traces, this typically updates the trace with final information.
        For spans and generations, this records the end time and completion data.

        Args:
            name: Optional updated name for the entity.
            user_id: Optional updated user ID for the entity.
            session_id: Optional updated session ID for the entity.
            input: Optional updated input data for the entity.
            output: Optional output/result data produced by the entity.
            metadata: Optional additional metadata to add to the entity.
            tags: Optional tags to add to the entity.
            timestamp: Optional timestamp for when the entity ended (defaults to now).

        Returns:
            StatefulObservabilityClient: The parent stateful client, allowing for continued
                method chaining after ending the current entity.

        Example:
            ```python
            # End a generation with its output
            await generation.end(
                output={"forecast": "Sunny, 25°C"},
                metadata={"completion_tokens": 42}
            )

            # End a trace with a final result
            await trace.end(
                output={"response": "The weather in Tokyo is sunny with 25°C"}
            )
            ```
        """
        ...

    @abc.abstractmethod
    async def flush(self) -> None:
        """
        Flush all pending events to ensure they are sent to the observability platform.

        This method ensures that all queued events are immediately processed and sent
        to the backend system. It's particularly important for short-lived applications
        like serverless functions where the process might terminate before background
        threads have a chance to send all events.

        The method is typically blocking and will wait until all events have been processed.

        Example:
            ```python
            # At the end of your application or before shutdown
            await client.flush()
            ```
        """
        ...

    @abc.abstractmethod
    async def score_trace(
        self,
        *,
        name: str,
        value: float | str,
        trace_id: str | None = None,
        comment: str | None = None,
    ) -> None:
        """
        Add a score to a trace.

        Scores provide quantitative or categorical assessments of trace quality or performance.
        They can be used for filtering, sorting, and analysis in the observability platform.

        Args:
            name: Identifier for the score (e.g., "trace_success", "response_quality").
            value: The score value, which can be:
                - A float (typically 0.0-1.0) for numeric scores
                - A string for categorical scores
            trace_id: Optional ID of the trace to score. If not provided, uses the current trace.
            comment: Optional explanation or context for the score.

        Example:
            ```python
            # Mark a trace as successful
            await client.score_trace(name="trace_success", value=1.0, comment="Completed without errors")

            # Categorize a trace
            await client.score_trace(name="response_type", value="detailed", comment="Comprehensive answer")
            ```
        """
        ...

    # Helper methods that build on the abstract methods

    async def model_generation(
        self,
        *,
        provider: str,
        model: str,
        input_data: dict[str, object],
        metadata: dict[str, object] | None = None,
        name: str | None = None,
    ) -> StatefulObservabilityClient:
        """
        Create a standardized generation trace for model invocations.

        A convenience method that creates a generation with standardized naming
        and common fields for AI model generations.

        Args:
            provider: The provider name (e.g., "google", "openai", "anthropic")
            model: The model identifier
            input_data: The input data sent to the model
            metadata: Additional metadata to track
            name: Optional custom name (defaults to "{provider}_{model}_generation")

        Returns:
            A new stateful client for the created generation
        """
        combined_metadata: dict[str, object] = {"provider": provider, "model": model}
        if metadata:
            combined_metadata.update(metadata)

        return await self.generation(
            name=name or f"{provider}_{model}_generation",
            input=input_data,
            metadata=combined_metadata,
        )

    async def complete_with_success(
        self,
        *,
        output: dict[str, object],
        start_time: datetime | None = None,
        metadata: dict[str, object] | None = None,
        usage_details: dict[str, object] | None = None,
        cost_details: dict[str, object] | None = None,
    ) -> StatefulObservabilityClient:
        """
        End the current trace/span/generation with success status and timing.

        Args:
            output: The output data
            start_time: Start time for calculating latency
            metadata: Additional metadata
            usage_details: Usage information (tokens, etc.)
            cost_details: Cost information

        Returns:
            The parent stateful client
        """
        complete_metadata: dict[str, object] = {"status": "success"}

        if start_time:
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            complete_metadata["latency_ms"] = latency_ms

        if metadata:
            complete_metadata.update(metadata)

        # Make sure that cost_details contains proper field names
        if cost_details and "total" in cost_details:
            # Ensure we have all necessary fields with consistent naming
            standardized_cost = {
                "input": cost_details.get("input", 0.0),
                "output": cost_details.get("output", 0.0),
                "total": cost_details.get("total", 0.0),
                "currency": cost_details.get("currency", "USD"),
            }
            cost_details = standardized_cost

        return await self.end(
            output=output,
            metadata=complete_metadata,
            usage_details=usage_details,
            cost_details=cost_details,
        )

    async def complete_with_error(
        self,
        *,
        error: Exception | str,
        start_time: datetime | None = None,
        error_type: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> StatefulObservabilityClient:
        """
        End the current trace/span/generation with error information.

        Args:
            error: The exception that occurred or error message
            start_time: Start time for calculating latency
            error_type: Type of error (defaults to the exception class name)
            metadata: Additional metadata

        Returns:
            The parent stateful client
        """
        complete_metadata: dict[str, object] = {"status": "error"}

        if isinstance(error, Exception):
            complete_metadata["error_type"] = error_type or type(error).__name__
            error_output: dict[str, object] = {"error": str(error)}
        else:
            complete_metadata["error_type"] = error_type or "Error"
            error_output = {"error": error}

        if start_time:
            # Convert float to object type for dictionary
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            complete_metadata["latency_ms"] = latency_ms

        if metadata:
            complete_metadata.update(metadata)

        return await self.end(output=error_output, metadata=complete_metadata)
