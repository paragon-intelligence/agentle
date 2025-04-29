"""
Model Preferences Module

This module defines the ModelPreference class used to specify preferences
for model selection and behavior during generation.
"""

from typing import Sequence

from agentle.mcp.sampling.hint import Hint
from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class ModelPreference(BaseModel):
    """
    Represents preferences for model selection and behavior.

    This class allows specifying various preferences that influence which model is
    selected for a generation task and how it should prioritize different aspects
    like cost efficiency, speed, and intelligence.

    Attributes:
        hints: Optional sequence of hints to guide model selection or behavior.
        costPriority: Optional priority value for cost efficiency (0.0 to 1.0).
        speedPriority: Optional priority value for generation speed (0.0 to 1.0).
        intelligencePriority: Optional priority value for model intelligence (0.0 to 1.0).
    """

    hints: Sequence[Hint] | None = Field(
        default=None,
        description="Optional list of hints to guide model selection or behavior",
        examples=[[{"name": "use_knowledge_base"}, {"name": "factual_response"}]],
    )
    costPriority: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Priority value for cost efficiency (0.0 to 1.0, where higher values indicate stronger preference for cost-efficient models)",
        examples=[0.8],
    )
    speedPriority: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Priority value for generation speed (0.0 to 1.0, where higher values indicate stronger preference for faster models)",
        examples=[0.5],
    )
    intelligencePriority: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Priority value for model intelligence (0.0 to 1.0, where higher values indicate stronger preference for more capable models)",
        examples=[0.9],
    )
