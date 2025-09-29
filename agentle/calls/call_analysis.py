from typing import Any
from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class CallAnalysis(BaseModel):
    """Post-call analysis results"""

    summary: str = Field(default="", description="Call summary")
    key_points: list[str] = Field(
        default_factory=list, description="Important conversation points"
    )
    action_items: list[str] = Field(
        default_factory=list, description="Identified action items"
    )
    sentiment_analysis: dict[str, Any] = Field(
        default_factory=dict, description="Detailed sentiment breakdown"
    )
    goal_achievement: dict[str, Any] = Field(
        default_factory=dict, description="Goal completion analysis"
    )
    custom_analysis: dict[str, Any] = Field(
        default_factory=dict, description="Custom analysis results"
    )
    analysis_model: str = Field(default="", description="Model used for analysis")
    processing_time_seconds: float = Field(
        default=0.0, ge=0.0, description="Analysis processing time"
    )
