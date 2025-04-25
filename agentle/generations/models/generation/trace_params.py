from collections.abc import Sequence
from typing import Any, NotRequired, TypedDict


class TraceParams(TypedDict, total=False):
    """Parameters for tracking and analyzing LLM interactions.

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
