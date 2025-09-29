from typing import Any

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.calls.schedule_plan import SchedulePlan
from agentle.calls.transcriber_config import TranscriberConfig
from agentle.calls.voice_config import VoiceConfig


class CallConfig(BaseModel):
    """Configuration for making a phone call"""

    # Core settings
    max_duration_seconds: int = Field(
        default=1800,
        ge=30,
        le=7200,
        description="Maximum call duration (30min default)",
    )
    voice_config: VoiceConfig | None = Field(
        default=None, description="Voice configuration override"
    )
    transcriber_config: TranscriberConfig | None = Field(
        default=None, description="STT configuration override"
    )

    # Assistant behavior overrides
    assistant_overrides: dict[str, Any] = Field(
        default_factory=dict, description="Runtime assistant modifications"
    )
    variable_values: dict[str, Any] = Field(
        default_factory=dict, description="Dynamic variable values"
    )

    # Call management
    schedule_plan: SchedulePlan | None = Field(
        default=None, description="Future call scheduling"
    )
    enable_recording: bool = Field(default=True, description="Record the call")
    enable_live_transcription: bool = Field(
        default=True, description="Provide real-time transcription"
    )

    # Advanced features
    enable_interruptions: bool = Field(
        default=True, description="Allow customer to interrupt assistant"
    )
    background_sound_reduction: bool = Field(
        default=True, description="Reduce background noise"
    )
    background_voice_filtering: bool = Field(
        default=True, description="Filter background voices"
    )

    # Integration settings
    webhook_url: str | None = Field(
        default=None, description="URL for real-time events"
    )
    webhook_secret: str | None = Field(
        default=None, description="Secret for webhook verification"
    )
    custom_headers: dict[str, str] = Field(
        default_factory=dict, description="Custom HTTP headers"
    )

    # Retry and error handling
    retry_on_failure: bool = Field(default=True, description="Retry failed calls")
    max_retries: int = Field(
        default=2, ge=0, le=5, description="Maximum retry attempts"
    )

    # Analytics
    enable_analysis: bool = Field(default=True, description="Enable post-call analysis")
    analysis_schema: dict[str, Any] | None = Field(
        default=None, description="Custom analysis schema"
    )
