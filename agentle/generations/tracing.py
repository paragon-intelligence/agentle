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
from datetime import datetime
from typing import Sequence


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
        metadata: dict[str, object] | None = None,
        tags: Sequence[str] | None = None,
        timestamp: datetime | None = None,
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
    ) -> StatefulObservabilityClient:
        """
        End the current trace/span/generation with success status and timing.

        Args:
            output: The output data
            start_time: Start time for calculating latency
            metadata: Additional metadata

        Returns:
            The parent stateful client
        """
        complete_metadata: dict[str, object] = {"status": "success"}

        if start_time:
            # Convert float to object type for dictionary
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            complete_metadata["latency_ms"] = latency_ms

        if metadata:
            complete_metadata.update(metadata)

        return await self.end(output=output, metadata=complete_metadata)

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
            error_output: dict[str, object] = {"error": error}

        if start_time:
            # Convert float to object type for dictionary
            latency_ms = (datetime.now() - start_time).total_seconds() * 1000
            complete_metadata["latency_ms"] = latency_ms

        if metadata:
            complete_metadata.update(metadata)

        return await self.end(output=error_output, metadata=complete_metadata)


class LangfuseObservabilityClient(StatefulObservabilityClient):
    """
    Implementation of StatefulObservabilityClient using Langfuse.

    This class connects the Agentle framework's observability interface to the Langfuse
    platform, enabling detailed tracking of AI model operations, usage patterns, and
    performance metrics.

    Langfuse-specific features are abstracted behind the common StatefulObservabilityClient
    interface, allowing applications to use Langfuse without direct dependencies on its API.
    The implementation handles the mapping between Agentle's tracing concepts and Langfuse's
    data model.

    The client can be initialized either with an existing Langfuse client, with a stateful
    Langfuse client to wrap (for creating hierarchical structures), or with default settings
    that use environment variables for configuration.

    Key features:
    - Hierarchical tracing with traces, spans, generations, and events
    - Method chaining for fluent interface
    - Integration with Langfuse's scoring and evaluation features
    - Support for metadata, tagging, and timing information

    Environment variables used (when no client is provided):
    - LANGFUSE_HOST (optional, defaults to Langfuse Cloud)
    - LANGFUSE_PUBLIC_KEY
    - LANGFUSE_SECRET_KEY
    - LANGFUSE_PROJECT (optional)

    Attributes:
        _client: The underlying Langfuse client
        _stateful_client: Optional stateful Langfuse client for hierarchical operations
        _trace_id: The current trace ID for this client instance
        _logger: Logger instance for this class

    Example:
        ```python
        # Basic initialization using environment variables
        client = LangfuseObservabilityClient()

        # Start a trace and track AI operations
        trace_client = await client.trace(name="process_request", user_id="user123")
        generation_client = await trace_client.generation(name="answer_generation", metadata={"model": "gemini-1.5-pro"})
        await generation_client.end(output={"text": "Generated response"})
        await trace_client.end()  # End the trace
        ```
    """

    _client: Langfuse
    _stateful_client: Optional[LangfuseStatefulClient]
    _trace_id: Optional[str]

    def __init__(
        self,
        client: Optional[Langfuse] = None,
        stateful_client: Optional[LangfuseStatefulClient] = None,
        trace_id: Optional[str] = None,
        secret_key: Optional[str] = None,
        public_key: Optional[str] = None,
        host: Optional[str] = None,
    ) -> None:
        """
        Initialize a new LangfuseObservabilityClient.

        Creates a new observability client connected to Langfuse. The client can be
        initialized in several ways:

        1. With default settings (using environment variables for authentication)
        2. With an existing Langfuse client
        3. With a stateful Langfuse client (usually from a parent operation)

        When no trace_id is provided, a random UUID is generated to ensure unique
        trace identification.

        Args:
            client: Optional existing Langfuse client to use. If not provided,
                a new client will be created using environment variables.
            stateful_client: Optional stateful Langfuse client to wrap. This is typically
                used internally when creating hierarchical structures through method chaining.
            trace_id: Optional trace ID to use. If not provided, a random UUID will be generated.
                This ensures that even standalone operations are properly tracked.

        Note:
            When creating a client with default settings, the following environment
            variables are used:
            - LANGFUSE_HOST (optional, defaults to Langfuse Cloud)
            - LANGFUSE_PUBLIC_KEY
            - LANGFUSE_SECRET_KEY
            - LANGFUSE_PROJECT (optional)

        Example:
            ```python
            # Create with default settings from environment variables
            default_client = LangfuseObservabilityClient()

            # Create with an existing Langfuse client
            from langfuse import Langfuse
            langfuse_client = Langfuse(
                host="https://cloud.langfuse.com",
                public_key="pk-lf-...",
                secret_key="sk-lf-..."
            )
            custom_client = LangfuseObservabilityClient(client=langfuse_client)
            ```
        """
        from langfuse.client import Langfuse

        self._logger = logging.getLogger(self.__class__.__name__)
        self._client = client or Langfuse(
            host=host,
            secret_key=secret_key,
            public_key=public_key,
        )
        self._stateful_client = stateful_client
        self._trace_id = trace_id or str(uuid.uuid4())

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
        """
        Create a trace in Langfuse.

        Creates a new trace in Langfuse, which represents a complete user interaction or
        system process. In Langfuse, a trace is the top-level container for observability
        data, potentially containing spans, generations, and events.

        The method returns a new stateful client that wraps the created trace, enabling
        method chaining for creating hierarchical structures.

        Args:
            name: Identifier of the trace. Should be descriptive of the operation.
            user_id: The id of the user that triggered the execution. Used for
                filtering and analyzing user-specific patterns.
            session_id: Used to group multiple traces into a session. Helpful for
                tracking multi-step interactions that span multiple traces.
            input: The input of the trace. Can be any data structure that triggered
                this operation.
            output: The output of the trace. Typically set later using end().
            metadata: Additional metadata for the trace. Can include any contextual
                information that might be useful for analysis.
            tags: Tags for categorizing the trace. Useful for filtering and grouping.
            timestamp: The timestamp of when the trace started. Defaults to now if not provided.

        Returns:
            A new LangfuseObservabilityClient instance wrapping the created trace.

        Example:
            ```python
            # Create a trace for a user query
            trace_client = await client.trace(
                name="answer_user_question",
                user_id="user123",
                input={"question": "How does AI work?"},
                metadata={"source": "chat_interface", "priority": "high"},
                tags=["question", "educational"]
            )
            ```
        """
        trace = self._client.trace(
            name=name,
            user_id=user_id,
            session_id=session_id,
            input=input,
            output=output,
            metadata=metadata,
            tags=list(tags) if tags else None,
            timestamp=timestamp,
        )

        return LangfuseObservabilityClient(
            client=self._client,
            stateful_client=trace,
            trace_id=trace.trace_id,
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
        """
        Create a generation in Langfuse.

        Creates a new generation in Langfuse, which represents a specific AI model invocation
        that produces content. In Langfuse, generations are specialized spans that contain
        additional LLM-specific fields like model, prompt, and completion information.

        The method returns a new stateful client that wraps the created generation,
        enabling method chaining for creating hierarchical structures.

        This method behaves differently depending on whether this client already has
        a stateful client (is part of a trace hierarchy):
        - If it has a stateful client, the generation is created as a child of that client
        - If not, it creates a standalone generation linked to the current trace_id

        Args:
            name: Identifier of the generation. Should describe what's being generated.
            user_id: The id of the user that triggered the generation.
            session_id: Used to group related generations into a session.
            input: The input/prompt for the generation. Typically the data sent to the model.
            output: The output of the generation. Typically set later using end().
            metadata: Additional metadata for the generation. Often includes model parameters
                like temperature, top_p, etc.
            tags: Tags for categorizing the generation.
            timestamp: The timestamp of when the generation started. Defaults to now.

        Returns:
            A new LangfuseObservabilityClient instance wrapping the created generation.

        Example:
            ```python
            # Create a generation for a text completion
            generation_client = await trace_client.generation(
                name="summary_generation",
                input={"text": "Summarize this article: [...]"},
                metadata={
                    "model": "gemini-1.5-pro",
                    "temperature": 0.7,
                    "max_tokens": 200
                }
            )
            ```
        """
        if self._stateful_client:
            # If we already have a stateful client, use it to create a generation
            generation = self._stateful_client.generation(
                name=name,
                input=input,
                output=output,
                metadata=metadata,
                start_time=timestamp,
            )
        else:
            # Otherwise, create a new generation directly
            generation = self._client.generation(
                trace_id=self._trace_id,
                name=name,
                input=input,
                output=output,
                metadata=metadata,
                start_time=timestamp,
            )

        return LangfuseObservabilityClient(
            client=self._client,
            stateful_client=generation,
            trace_id=generation.trace_id,
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
        """
        Create a span in Langfuse.

        Creates a new span in Langfuse, which represents a subtask or phase within a
        larger operation. Spans are useful for breaking down complex operations into
        smaller, measurable units that can be analyzed independently.

        The method returns a new stateful client that wraps the created span,
        enabling method chaining for creating hierarchical structures.

        This method behaves differently depending on whether this client already has
        a stateful client (is part of a trace hierarchy):
        - If it has a stateful client, the span is created as a child of that client
        - If not, it creates a standalone span linked to the current trace_id

        Args:
            name: Identifier of the span. Should describe the subtask or phase.
            user_id: The id of the user related to this span.
            session_id: Used to group related spans into a session.
            input: The input to the span. Typically the data being processed.
            output: The output of the span. Typically set later using end().
            metadata: Additional metadata for the span. Can include any relevant
                contextual information.
            tags: Tags for categorizing the span.
            timestamp: The timestamp of when the span started. Defaults to now.

        Returns:
            A new LangfuseObservabilityClient instance wrapping the created span.

        Example:
            ```python
            # Create a span for data processing
            span_client = await trace_client.span(
                name="extract_keywords",
                input={"text": "Machine learning is transforming industry..."},
                metadata={"algorithm": "TF-IDF", "max_keywords": 10}
            )
            ```
        """
        if self._stateful_client:
            # If we already have a stateful client, use it to create a span
            span = self._stateful_client.span(
                name=name,
                input=input,
                output=output,
                metadata=metadata,
                start_time=timestamp,
            )
        else:
            # Otherwise, create a new span directly
            span = self._client.span(
                trace_id=self._trace_id,
                name=name,
                input=input,
                output=output,
                metadata=metadata,
                start_time=timestamp,
            )

        return LangfuseObservabilityClient(
            client=self._client,
            stateful_client=span,
            trace_id=span.trace_id,
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
        """
        Create an event in Langfuse.

        Creates a new event in Langfuse, which represents a discrete point of interest
        within a trace. Events are useful for marking significant moments or decisions
        that don't have a duration but are important to track.

        The method returns a new stateful client that wraps the created event,
        enabling method chaining for further operations.

        This method behaves differently depending on whether this client already has
        a stateful client (is part of a trace hierarchy):
        - If it has a stateful client, the event is created as a child of that client
        - If not, it creates a standalone event linked to the current trace_id

        Args:
            name: Identifier of the event. Should describe what occurred.
            user_id: The id of the user related to this event.
            session_id: Used to group related events into a session.
            input: Input data related to this event.
            output: Output data related to this event.
            metadata: Additional metadata for the event. Can include any relevant
                contextual information.
            tags: Tags for categorizing the event.
            timestamp: The timestamp of when the event occurred. Defaults to now.

        Returns:
            A new LangfuseObservabilityClient instance wrapping the created event.

        Example:
            ```python
            # Create an event for a threshold exceeded
            event_client = await span_client.event(
                name="quota_exceeded",
                metadata={"limit": 1000, "current_usage": 1001, "user_notified": True}
            )
            ```
        """
        if self._stateful_client:
            # If we already have a stateful client, use it to create an event
            event = self._stateful_client.event(
                name=name,
                input=input,
                output=output,
                metadata=metadata,
                start_time=timestamp,
            )
        else:
            # Otherwise, create a new event directly
            event = self._client.event(
                trace_id=self._trace_id,
                name=name,
                input=input,
                output=output,
                metadata=metadata,
                start_time=timestamp,
            )

        return LangfuseObservabilityClient(
            client=self._client,
            stateful_client=event,
            trace_id=event.trace_id,
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
        metadata: dict[str, object] | None = None,
        tags: Sequence[str] | None = None,
        timestamp: datetime | None = None,
    ) -> StatefulObservabilityClient:
        """
        End the current observation in Langfuse.

        Marks the current entity (trace, span, or generation) as complete and optionally
        updates it with final information. The behavior depends on the type of entity:

        - For spans and generations: Calls the end() method with the provided data
        - For traces: Calls the update() method with the provided data

        This method is essential for properly completing the observability lifecycle
        and ensuring accurate duration measurements in Langfuse.

        Args:
            name: Optional updated name for the entity.
            user_id: Optional updated user ID for the entity (traces only).
            session_id: Optional updated session ID for the entity (traces only).
            input: Optional updated input data for the entity.
            output: Optional output/result data produced by the entity.
            metadata: Optional additional metadata to add to the entity.
            tags: Optional tags to add to the entity (traces only).
            timestamp: Optional timestamp for when the entity ended. Defaults to now.

        Returns:
            The same stateful client for method chaining. This allows for ending
            nested entities in sequence when using method chaining.

        Example:
            ```python
            # End a generation with its result
            await generation_client.end(
                output={"text": "Generated response about the topic..."},
                metadata={"tokens": 156, "completion_time_ms": 450}
            )

            # End a trace with a final summary
            await trace_client.end(
                output={"final_response": "The processed result of the entire operation"},
                metadata={"success": True, "total_time_ms": 1250}
            )
            ```
        """
        from langfuse.client import (
            StatefulGenerationClient,
            StatefulSpanClient,
            StatefulTraceClient,
        )

        if self._stateful_client:
            if isinstance(
                self._stateful_client,
                (StatefulSpanClient, StatefulGenerationClient),
            ):
                # For spans and generations, call end()
                self._stateful_client.end(
                    name=name,
                    input=input,
                    output=output,
                    metadata=metadata,
                    end_time=timestamp,
                )
            elif isinstance(self._stateful_client, StatefulTraceClient):
                # For traces, call update()
                self._stateful_client.update(
                    name=name,
                    user_id=user_id,
                    session_id=session_id,
                    input=input,
                    output=output,
                    metadata=metadata,
                    tags=list(tags) if tags else None,
                )

        return self

    async def flush(self) -> None:
        """
        Flush all pending events to Langfuse.

        This method ensures that all queued events are sent to the Langfuse backend
        before the application exits. It's especially important for short-lived
        applications (like serverless functions) where the process might terminate
        before the background thread has a chance to send all events.

        The method is blocking and will wait until all events have been processed.

        Example:
            ```python
            # At the end of your application or before shutdown
            await client.flush()
            ```
        """
        try:
            self._client.flush()
            self._logger.debug("Successfully flushed all events to Langfuse")
        except Exception as e:
            self._logger.error(f"Error flushing events to Langfuse: {e}")


class TracingManager:
    """
    Manager for tracing AI generation activities across all providers.

    This class encapsulates the logic for creating, managing, and completing traces,
    providing a consistent interface for all generation providers to use. It handles
    the complexities of trace hierarchy, error handling, and event flushing.

    Attributes:
        tracing_client: The client used for observability and tracing.
        provider_name: The name of the provider using this tracing manager.
    """

    def __init__(
        self,
        tracing_client: Optional[StatefulObservabilityClient] = None,
        provider_name: str = "unknown",
    ) -> None:
        """
        Initialize a new tracing manager.

        Args:
            tracing_client: Optional client for observability and tracing.
            provider_name: The name of the provider using this tracing manager.
        """
        self.tracing_client = tracing_client
        self.provider_name = provider_name

    async def setup_trace(
        self,
        *,
        generation_config: GenerationConfig,
        model: str,
        input_data: dict[str, Any],
        is_final_generation: bool = False,
    ) -> tuple[
        Optional[StatefulObservabilityClient], Optional[StatefulObservabilityClient]
    ]:
        """
        Set up tracing for a generation by creating or retrieving a trace.

        This method handles the logic for determining whether to create a new trace
        or reuse an existing one, based on the trace_params in the generation_config.

        Args:
            generation_config: The configuration for the generation.
            model: The model being used for generation.
            input_data: The input data for the generation.
            is_final_generation: Whether this is the final generation in a sequence.

        Returns:
            A tuple containing (trace_client, generation_client) for tracing, both may be None.
        """
        if not self.tracing_client:
            return None, None

        trace_params = generation_config.trace_params
        user_id = trace_params.get("user_id", "anonymous")
        session_id = trace_params.get("session_id")

        # Get or create conversation trace
        trace_client = None
        parent_trace_id = trace_params.get("parent_trace_id")

        # Try to get existing trace or create new one
        try:
            if parent_trace_id:
                # Use existing trace ID with parent_trace_id
                trace_client = await self.tracing_client.trace(
                    name=trace_params.get("name"),
                    user_id=user_id,
                    session_id=session_id,
                )
            else:
                # Create new trace
                trace_name = trace_params.get(
                    "name", f"{self.provider_name}_{model}_conversation"
                )

                trace_client = await self.tracing_client.trace(
                    name=trace_name,
                    user_id=user_id,
                    session_id=session_id,
                    input=input_data.get(
                        "trace_input",
                        {
                            "model": model,
                            "message_count": input_data.get("message_count", 0),
                            "has_tools": input_data.get("has_tools", False),
                            "has_schema": input_data.get("has_schema", False),
                        },
                    ),
                    metadata={
                        "provider": self.provider_name,
                        "model": model,
                    },
                )

                # Store trace_id for future calls if not final generation
                if trace_client and not is_final_generation:
                    # Check if trace_client has an id attribute and access it safely
                    trace_id = self._get_trace_id(trace_client)
                    if trace_id:
                        trace_params["parent_trace_id"] = trace_id
        except Exception:
            # Fall back to no tracing if we encounter errors
            trace_client = None

        # Set up generation tracing
        generation_client = None
        if trace_client:
            # Get trace metadata
            trace_metadata: dict[str, Any] = {}
            if "metadata" in trace_params and isinstance(
                trace_params["metadata"], dict
            ):
                trace_metadata = {
                    k: v
                    for k, v in trace_params["metadata"].items()
                    if isinstance(k, str)
                }

            # Create generation
            try:
                generation_name = trace_params.get(
                    "name", f"{self.provider_name}_{model}_generation"
                )

                # Extract config metadata if available
                config_data = {}
                if generation_config:
                    config_data = {
                        k: v
                        for k, v in generation_config.__dict__.items()
                        if not k.startswith("_") and not callable(v)
                    }

                generation_client = await trace_client.model_generation(
                    provider=self.provider_name,
                    model=model,
                    input_data=input_data,
                    metadata={
                        "config": config_data,
                        **trace_metadata,
                    },
                    name=generation_name,
                )
            except Exception:
                # Fall back to no generation tracing if errors
                generation_client = None

        return trace_client, generation_client

    def _get_trace_id(self, trace_client: StatefulObservabilityClient) -> Optional[str]:
        """
        Safely get the ID from a trace client.

        Args:
            trace_client: The trace client to get the ID from.

        Returns:
            The trace ID if available, otherwise None.
        """
        # Handle trace clients that use different ID attributes
        if hasattr(trace_client, "id"):
            return cast(str, getattr(trace_client, "id"))
        elif hasattr(trace_client, "trace_id"):
            return cast(str, getattr(trace_client, "trace_id"))
        return None

    async def complete_generation(
        self,
        *,
        generation_client: Optional[StatefulObservabilityClient],
        start_time: datetime,
        output_data: dict[str, Any],
        trace_metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Complete a generation with success.

        Args:
            generation_client: The client for the generation to complete.
            start_time: When the generation started.
            output_data: The data produced by the generation.
            trace_metadata: Additional metadata for the trace.
        """
        if generation_client:
            await generation_client.complete_with_success(
                output=output_data,
                start_time=start_time,
                metadata=trace_metadata or {},
            )

    async def complete_trace(
        self,
        *,
        trace_client: Optional[StatefulObservabilityClient],
        generation_config: GenerationConfig,
        output_data: dict[str, Any],
        success: bool = True,
    ) -> None:
        """
        Complete a trace with success or error.

        This method handles the logic for properly completing a trace and ensuring
        that all events are flushed to the tracing client.

        Args:
            trace_client: The client for the trace to complete.
            generation_config: The configuration used for generation.
            output_data: The data produced by the generation.
            success: Whether the operation was successful.
        """
        if not trace_client:
            return

        try:
            # Complete the trace
            await trace_client.end(
                output=output_data,
                metadata={"completion_status": "success" if success else "error"},
            )

            # Flush events and clean up
            if self.tracing_client:
                await self.tracing_client.flush()

            # Clean up trace_params
            trace_params = generation_config.trace_params
            if "parent_trace_id" in trace_params:
                del trace_params["parent_trace_id"]
        except Exception:
            # Just continue even if we can't clean up properly
            pass

    async def handle_error(
        self,
        *,
        generation_client: Optional[StatefulObservabilityClient],
        trace_client: Optional[StatefulObservabilityClient],
        generation_config: GenerationConfig,
        start_time: datetime,
        error: Exception,
        trace_metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Handle an error during generation.

        This method records the error with the tracing client and ensures proper cleanup.

        Args:
            generation_client: The client for the generation that failed.
            trace_client: The client for the trace containing the generation.
            generation_config: The configuration used for generation.
            start_time: When the generation started.
            error: The exception that occurred.
            trace_metadata: Additional metadata for the trace.
        """
        error_str = str(error) if error else "Unknown error"

        # Complete generation with error
        if generation_client:
            await generation_client.complete_with_error(
                error=error_str,
                start_time=start_time,
                error_type="Exception",
                metadata=trace_metadata or {},
            )

        # Complete the trace with error
        await self.complete_trace(
            trace_client=trace_client,
            generation_config=generation_config,
            output_data={"error": error_str},
            success=False,
        )
