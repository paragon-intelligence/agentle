from typing import Sequence

from pydantic import BaseModel, Field

from agentle.generations.models.chain_of_thought.thought_detail import ThoughtDetail


class Step(BaseModel):
    """A step in a chain of thought.

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
        # examples=[1, 2, 3],
    )

    explanation: str = Field(
        description="A concise description of what was done in this step.",
        # examples=["Analyze the input statement", "Apply the quadratic formula"],
    )

    details: Sequence[ThoughtDetail] = Field(
        description="A list of specific details for each step in the reasoning.",
        # examples=[
        #     [
        #         {"detail": "Check initial values"},
        #         {"detail": "Confirm there are no inconsistencies"},
        #     ]
        # ],
    )
