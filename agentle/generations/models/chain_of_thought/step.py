"""
Step model for Chain of Thought reasoning in the Agentle framework.

This module defines the Step component of the Chain of Thought reasoning structure.
Steps represent discrete phases in a reasoning process, sitting between the high-level
ChainOfThought model and the granular ThoughtDetail components.

Each Step provides:
- A numerical position in the reasoning sequence
- A concise explanation of what occurs in this phase of reasoning
- A collection of detailed observations, calculations, or considerations

Steps are essential for breaking down complex reasoning into manageable,
sequential components that can be individually examined, validated, and understood.
"""

from typing import Sequence

from pydantic import BaseModel, Field

from agentle.generations.models.chain_of_thought.thought_detail import ThoughtDetail


class Step(BaseModel):
    """
    A single step in a chain of thought reasoning process.

    Each Step represents a distinct phase in a logical reasoning sequence,
    containing both a high-level explanation of what was done in this step
    and a collection of more granular details that elaborate on specific
    aspects of the reasoning.

    Steps are numbered to maintain a clear sequence of the reasoning process,
    allowing for proper ordering when presented to users or when analyzing
    the reasoning path.

    In complex reasoning tasks, having explicit steps helps in:
    - Breaking down complex problems into manageable parts
    - Identifying exactly where reasoning might go astray
    - Providing visibility into the full logical progression
    - Making the overall reasoning process more understandable

    Attributes:
        step_number: The position of this step in the overall chain of thought
        explanation: A concise description of what was done in this step
        details: A list of specific details for each step in the reasoning

    Example:
        >>> Step(
        ...     step_number=1,
        ...     explanation="Analyze the input statement",
        ...     details=[
        ...         ThoughtDetail(detail="Check initial values"),
        ...         ThoughtDetail(detail="Confirm there are no inconsistencies")
        ...     ]
        ... )
    """

    step_number: int = Field(
        description="The position of this step in the overall chain of thought.",
    )

    explanation: str = Field(
        description="A concise description of what was done in this step.",
    )

    details: Sequence[ThoughtDetail] = Field(
        description="A list of specific details for each step in the reasoning.",
    )
