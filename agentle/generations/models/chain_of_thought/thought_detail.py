"""
Granular thought detail model for Chain of Thought reasoning.

This module defines the most atomic unit of reasoning in the Chain of Thought
framework - individual thought details that make up steps in a reasoning process.
These details provide specific insights, calculations, observations, or sub-conclusions
that support the overall step they belong to.
"""

from pydantic import BaseModel, Field


class ThoughtDetail(BaseModel):
    """
    A detailed explanation of a specific aspect of a reasoning step.

    ThoughtDetail represents the most granular level of reasoning in the Chain of Thought
    framework. Each detail captures a single observation, calculation, inference, or
    consideration that contributes to the reasoning step it belongs to.

    This granularity serves several purposes:
    - Allows precise examination of each atomic reasoning component
    - Makes complex reasoning steps more digestible through decomposition
    - Provides clear traceability of how each consideration affects the reasoning
    - Enables targeted feedback or correction at the most specific level

    Think of ThoughtDetails as the "atomic units" of a reasoning process that,
    when combined, form Steps, which in turn form a complete Chain of Thought.

    Attributes:
        detail: A granular explanation of a specific aspect of the reasoning step

    Example:
        >>> ThoughtDetail(detail="First, I added 2 + 3")
        >>> ThoughtDetail(detail="The velocity must be constant because acceleration is zero")
    """

    detail: str = Field(
        description="A granular explanation of a specific aspect of the reasoning step.",
        # examples=["First, I added 2 + 3", "Checked if the number is even or odd"],
    )
