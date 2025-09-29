from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class CallMetrics(BaseModel):
    """Call performance and analytics metrics"""

    # Duration metrics
    total_duration_seconds: float = Field(ge=0.0, description="Total call duration")
    conversation_duration_seconds: float = Field(
        ge=0.0, description="Active conversation time"
    )
    setup_duration_seconds: float = Field(ge=0.0, description="Time to establish call")

    # Audio quality metrics
    audio_quality_score: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Overall audio quality"
    )
    interruption_count: int = Field(
        default=0, ge=0, description="Number of conversation interruptions"
    )
    silence_duration_seconds: float = Field(
        default=0.0, ge=0.0, description="Total silence time"
    )

    # Cost breakdown
    total_cost_usd: float | None = Field(
        default=None, ge=0.0, description="Total call cost"
    )
    stt_cost_usd: float | None = Field(
        default=None, ge=0.0, description="Speech-to-text cost"
    )
    llm_cost_usd: float | None = Field(
        default=None, ge=0.0, description="Language model cost"
    )
    tts_cost_usd: float | None = Field(
        default=None, ge=0.0, description="Text-to-speech cost"
    )
    transport_cost_usd: float | None = Field(
        default=None, ge=0.0, description="Phone transport cost"
    )

    # Token usage
    total_tokens: int = Field(default=0, ge=0, description="Total tokens processed")
    prompt_tokens: int = Field(default=0, ge=0, description="Input tokens")
    completion_tokens: int = Field(default=0, ge=0, description="Output tokens")

    # Conversation metrics
    customer_satisfaction_score: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Estimated satisfaction"
    )
    goal_completion_rate: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Objective completion rate"
    )
    sentiment_score: float | None = Field(
        default=None, ge=-1.0, le=1.0, description="Overall conversation sentiment"
    )
