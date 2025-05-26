"""
Langfuse implementation of the StatefulObservabilityClient for Agentle.

This module provides the LangfuseObservabilityClient class, which implements the
StatefulObservabilityClient interface using Langfuse as the underlying observability platform.
Langfuse is a specialized observability platform for LLM applications that provides features
for tracing, logging, evaluating, and monitoring AI systems.

The implementation allows Agentle applications to send structured telemetry data to Langfuse,
enabling detailed visibility into AI model performance, usage patterns, costs, and quality.

This client supports the full StatefulObservabilityClient interface, including hierarchical
traces, spans, generations, and events, mapping these concepts to their Langfuse equivalents.

Example:
```python
from agentle.generations.tracing.langfuse import LangfuseObservabilityClient

# Create a client (uses environment variables for configuration)
client = LangfuseObservabilityClient()

# Track an AI-powered interaction
trace = await client.trace(
    name="user_conversation",
    user_id="user123",
    metadata={"session_type": "customer_support"}
)

# Log a model generation within the trace
generation = await trace.generation(
    name="initial_response",
    input={"query": "How do I reset my password?"},
    metadata={"model": "gemini-1.5-pro", "temperature": 0.7}
)

# Complete the generation with its result
await generation.end(
    output={"response": "You can reset your password by..."},
    metadata={"tokens": 42, "latency_ms": 850}
)

# Complete the trace
await trace.end()
```

Note: To use this client, you need to set up Langfuse credentials as environment variables:
- LANGFUSE_HOST (optional, defaults to Langfuse Cloud)
- LANGFUSE_PUBLIC_KEY
- LANGFUSE_SECRET_KEY
- LANGFUSE_PROJECT (optional)
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional, Sequence, override

from agentle.generations.tracing.contracts.stateful_observability_client import (
    StatefulObservabilityClient,
)

if TYPE_CHECKING:
    from langfuse import Langfuse
    from langfuse.client import StatefulClient as LangfuseStatefulClient


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
        metadata: Mapping[str, object] | None = None,
        tags: Sequence[str] | None = None,
        timestamp: datetime | None = None,
        usage_details: Mapping[str, Any] | None = None,
        cost_details: Mapping[str, Any] | None = None,
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

        if cost_details and "currency" in cost_details:
            cost_details = {k: v for k, v in cost_details.items() if k != "currency"}

        if self._stateful_client:
            if isinstance(
                self._stateful_client,
                (StatefulSpanClient, StatefulGenerationClient),
            ):
                # For spans and generations, call end()
                kwargs = {
                    "name": name,
                    "input": input,
                    "output": output,
                    "metadata": metadata,
                    "end_time": timestamp,
                }

                # Only add usage_details and cost_details if they're provided
                if usage_details:
                    kwargs["usage"] = usage_details
                if cost_details:
                    kwargs["cost_details"] = cost_details

                self._stateful_client.end(**kwargs)  # type: ignore
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

    @override
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

        This method adds a score to a trace in Langfuse, which can be used for
        evaluating and filtering traces in the UI.

        Args:
            name: The name of the score (e.g., "trace_success", "response_quality").
            value: The score value (float for numeric scores, string for categorical).
            trace_id: Optional trace ID. If not provided, uses the current trace.
            comment: Optional comment or explanation for the score.
        """
        try:
            target_trace_id = trace_id or self._trace_id

            if not target_trace_id:
                self._logger.warning("Cannot add score: No trace ID available")
                return

            if self._client:
                self._client.score(
                    trace_id=target_trace_id, name=name, value=value, comment=comment
                )
                self._logger.debug(
                    f"Added score '{name}' with value '{value}' to trace {target_trace_id}"
                )
        except Exception as e:
            self._logger.error(f"Error adding score to trace: {e}")
