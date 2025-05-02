"""
A2A Artifact Model

This module defines the Artifact class, which represents output artifacts created by agents
in the A2A protocol. Artifacts are structured outputs that can be generated during task
processing, such as code snippets, data analyses, or other content.
"""

from collections.abc import Sequence
from typing import Any

from rsb.models.base_model import BaseModel
from rsb.models.config_dict import ConfigDict
from rsb.models.field import Field

from agentle.generations.models.message_parts.text import TextPart


class Artifact(BaseModel):
    """
    Represents an output artifact created by an agent.

    Artifacts are structured outputs that can be generated during task processing,
    such as code snippets, data analyses, or other content. They can have names,
    descriptions, and may be generated in chunks (for streaming purposes).

    Attributes:
        name: Optional name of the artifact
        description: Optional description of the artifact
        parts: Sequence of text parts comprising the artifact content
        metadata: Optional additional metadata associated with the artifact
        index: Positional index of the artifact in the list of all artifacts
        append: Optional flag indicating if this artifact should be appended to an existing one
        last_chunk: Optional flag indicating if this is the last chunk of a streamed artifact

    Example:
        ```python
        from agentle.agents.a2a.models.artifact import Artifact
        from agentle.generations.models.message_parts.text import TextPart

        # Create a simple artifact
        code_artifact = Artifact(
            name="solution.py",
            description="Python solution for the problem",
            parts=[TextPart(text="def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a")],
            index=0
        )

        # Create a streaming artifact (first chunk)
        first_chunk = Artifact(
            name="large_result.txt",
            parts=[TextPart(text="This is the beginning of a large result...")],
            index=0,
            append=False,
            last_chunk=False
        )

        # Second chunk of the streaming artifact
        second_chunk = Artifact(
            name="large_result.txt",
            parts=[TextPart(text="...this is the continuation...")],
            index=0,
            append=True,
            last_chunk=False
        )

        # Final chunk of the streaming artifact
        final_chunk = Artifact(
            name="large_result.txt",
            parts=[TextPart(text="...and this is the end.")],
            index=0,
            append=True,
            last_chunk=True
        )
        ```

    Note:
        The Artifact model is immutable (frozen) once created, which means its
        attributes cannot be modified after initialization.
    """

    name: str | None = Field(default=None)
    """Optional name of the artifact"""

    description: str | None = Field(default=None)
    """Optional description of the artifact"""

    parts: Sequence[TextPart]
    """Sequence of text parts comprising the artifact content"""

    metadata: dict[str, Any] | None = Field(default=None)
    """Optional additional metadata associated with the artifact"""

    index: int
    """Positional index of the artifact in the list of all artifacts"""

    append: bool | None = Field(default=None)
    """Optional flag indicating if this artifact should be appended to an existing one"""

    last_chunk: bool | None = Field(default=None)
    """Optional flag indicating if this is the last chunk of a streamed artifact"""

    model_config = ConfigDict(frozen=True)
