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
        metadata: Custom JSON-serializable metadata
        tags: Categorization labels for filtering
        public: Visibility flag for trace data
        parent_trace_id: ID of parent trace for establishing trace hierarchy

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
    metadata: NotRequired[Any]
    tags: NotRequired[Sequence[str]]
    public: NotRequired[bool]
    parent_trace_id: NotRequired[str]
