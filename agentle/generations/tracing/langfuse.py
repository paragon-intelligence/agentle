from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional, Sequence, override

from agentle.generations.tracing.contracts.stateful_observability_client import (
    StatefulObservabilityClient,
)

if TYPE_CHECKING:
    from langfuse.client import StatefulGenerationClient
    from langfuse.client import StatefulSpanClient
    from langfuse.client import StatefulTraceClient
    from langfuse import Langfuse
    from langfuse.client import StatefulClient as LangfuseStatefulClient


class LangfuseObservabilityClient(StatefulObservabilityClient):
    """Implementation of StatefulObservabilityClient using Langfuse."""

    _client: Langfuse
    _stateful_client: Optional[LangfuseStatefulClient]
    _trace_id: Optional[str]

    def __init__(
        self,
        client: Optional[Langfuse] = None,
        stateful_client: Optional[LangfuseStatefulClient] = None,
        trace_id: Optional[str] = None,
    ) -> None:
        """Initialize a new LangfuseObservabilityClient.

        Args:
            client: The Langfuse client to use
            stateful_client: A stateful Langfuse client to wrap
            trace_id: The trace ID to use for new traces
        """
        from langfuse import Langfuse

        self._logger = logging.getLogger(self.__class__.__name__)
        self._client = client or Langfuse()
        self._stateful_client = stateful_client
        self._trace_id = trace_id or str(uuid.uuid4())

    @override
    def trace(
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
        """Create a trace in Langfuse.

        Args:
            name: Identifier of the trace
            user_id: The id of the user that triggered the execution
            session_id: Used to group multiple traces into a session
            input: The input of the trace
            output: The output of the trace
            metadata: Additional metadata of the trace
            tags: Tags for categorizing the trace
            timestamp: The timestamp of the trace

        Returns:
            A new stateful client for the created trace
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
    def generation(
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
        """Create a generation in Langfuse.

        Args:
            name: Identifier of the generation
            user_id: The id of the user that triggered the execution
            session_id: Used to group multiple generations into a session
            input: The input to the generation
            output: The output of the generation
            metadata: Additional metadata of the generation
            tags: Tags for categorizing the generation
            timestamp: The timestamp of the generation

        Returns:
            A new stateful client for the created generation
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
    def span(
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
        """Create a span in Langfuse.

        Args:
            name: Identifier of the span
            user_id: The id of the user that triggered the execution
            session_id: Used to group multiple spans into a session
            input: The input to the span
            output: The output of the span
            metadata: Additional metadata of the span
            tags: Tags for categorizing the span
            timestamp: The timestamp of the span

        Returns:
            A new stateful client for the created span
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
    def event(
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
        """Create an event in Langfuse.

        Args:
            name: Identifier of the event
            user_id: The id of the user that triggered the execution
            session_id: Used to group multiple events into a session
            input: The input to the event
            output: The output of the event
            metadata: Additional metadata of the event
            tags: Tags for categorizing the event
            timestamp: The timestamp of the event

        Returns:
            A new stateful client for the created event
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
    def end(
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
        """End the current observation in Langfuse.

        Args:
            name: Name for the ended observation
            user_id: User ID for the ended observation
            session_id: Session ID for the ended observation
            input: Input for the ended observation
            output: Output for the ended observation
            metadata: Metadata for the ended observation
            tags: Tags for the ended observation
            timestamp: Timestamp for the ended observation

        Returns:
            The same stateful client for chaining
        """
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
