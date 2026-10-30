"""
Tracing and observability parameters for AI generations in the Agentle framework.

This module defines the TraceParams TypedDict that encapsulates metadata and
configuration for tracing AI generations. Tracing provides observability into
generation requests and responses, enabling monitoring, debugging, analytics,
and audit capabilities throughout the system.

The parameters in this module allow for flexible configuration of what data is
captured, how it's identified, and what metadata is associated with a trace,
supporting diverse use cases from debugging to compliance requirements.
"""

from collections.abc import Sequence
from typing import Any, NotRequired, TypedDict


class TraceParams(TypedDict, total=False):
    """Parameters for tracking and analyzing LLM interactions.

    Traces provide a way to capture and analyze AI model interactions for
    purposes such as monitoring, debugging, analytics, and compliance.
    These parameters control what information is captured in a trace and
    how it's identified and categorized.

    All fields are optional, allowing for flexible configuration based on
    specific tracing needs and requirements.

    Attributes:
        name: Unique identifier for the trace
        input: Input parameters for the traced operation
        output: Result of the traced operation
        user_id: ID of user initiating the request
        session_id: Grouping identifier for related traces
        version: Version of the trace. Can be used for tracking changes
        release: Deployment release identifier
        environment: Deployment environment (e.g. production, staging)
        metadata: Custom JSON-serializable metadata
        tags: Categorization labels for filtering
        public: Visibility flag for trace data
        trace_id: Provider trace identifier
        trace_name: Provider trace display name
        span_name: Provider span display name
        generation_name: Provider generation display name
        parent_trace_id: ID of parent trace for establishing trace hierarchy
        parent_span_id: ID of parent span for provider-specific span hierarchy

    Example:
        >>> trace = TraceParams(
        ...     name="customer_support",
        ...     tags=["urgent", "billing"]
        ... )
    """

    name: NotRequired[str]
    input: NotRequired[Any]
    output: NotRequired[Any]
    user_id: NotRequired[str]
    session_id: NotRequired[str]
    version: NotRequired[str]
    release: NotRequired[str]
    environment: NotRequired[str]
    metadata: NotRequired[dict[str, Any]]
    tags: NotRequired[Sequence[str]]
    public: NotRequired[bool]
    trace_id: NotRequired[str]
    trace_name: NotRequired[str]
    span_name: NotRequired[str]
    generation_name: NotRequired[str]
    parent_trace_id: NotRequired[str]
    parent_span_id: NotRequired[str]
