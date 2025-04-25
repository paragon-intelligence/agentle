from pydantic import BaseModel, Field


class ThoughtDetail(BaseModel):
    """A detailed explanation of a specific aspect of a reasoning step.

    Attributes:
        detail: A granular explanation of a specific aspect of the reasoning step

    Example:
        >>> ThoughtDetail(detail="First, I added 2 + 3")
    """

    detail: str = Field(
        description="A granular explanation of a specific aspect of the reasoning step.",
        # examples=["First, I added 2 + 3", "Checked if the number is even or odd"],
    )
