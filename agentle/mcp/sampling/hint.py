"""
Model Hint Module

This module defines the Hint class used to provide guidance or suggestions
to the model selection and sampling process.
"""

from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class Hint(BaseModel):
    """
    Represents a hint or suggestion for model selection or behavior.

    Hints are lightweight indicators that can influence how models are selected
    or how they operate during generation without strictly enforcing requirements.

    Attributes:
        name: Optional name of the hint to provide to the model or selection system.
    """

    name: str | None = Field(
        default=None,
        description="Optional name identifying the hint type",
        examples=["use_system_context", "enable_creativity", "factual_response"],
    )
