"""
Google-specific generation configuration.

This module keeps provider-specific GenerateContentConfig options out of the
provider-neutral GenerationConfig field list while still allowing agents to pass
Google knobs through the existing generationConfig path.
"""

from __future__ import annotations

from typing import Any

from rsb.models.base_model import BaseModel
from rsb.models.config_dict import ConfigDict
from rsb.models.field import Field


class GoogleGenerationConfig(BaseModel):
    """
    Google-only options forwarded to google.genai.types.GenerateContentConfig.
    """

    automatic_function_calling: Any | None = Field(default=None)
    cached_content: str | None = Field(default=None)
    audio_timestamp: bool | None = Field(default=None)
    enable_enhanced_civic_answers: bool | None = Field(default=None)
    http_options: Any | None = Field(default=None)
    image_config: Any | None = Field(default=None)
    labels: dict[str, str] | None = Field(default=None)
    media_resolution: Any | None = Field(default=None)
    model_armor_config: Any | None = Field(default=None)
    model_selection_config: Any | None = Field(default=None)
    response_json_schema: Any | None = Field(default=None)
    response_mime_type: str | None = Field(default=None)
    response_modalities: list[str] | None = Field(default=None)
    routing_config: Any | None = Field(default=None)
    safety_settings: list[Any] | None = Field(default=None)
    service_tier: str | None = Field(default=None)
    should_return_http_response: bool | None = Field(default=None)
    speech_config: Any | None = Field(default=None)
    thinking_config: Any | None = Field(default=None)
    tool_config: Any | None = Field(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True)
