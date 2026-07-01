"""
TypedDict counterpart for Google-specific generation configuration.
"""

from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class GoogleGenerationConfigDict(TypedDict):
    """
    Dictionary form of Google-only GenerateContentConfig options.
    """

    automatic_function_calling: NotRequired[Any | None]
    cached_content: NotRequired[str | None]
    audio_timestamp: NotRequired[bool | None]
    enable_enhanced_civic_answers: NotRequired[bool | None]
    http_options: NotRequired[Any | None]
    image_config: NotRequired[Any | None]
    labels: NotRequired[dict[str, str] | None]
    media_resolution: NotRequired[Any | None]
    model_armor_config: NotRequired[Any | None]
    model_selection_config: NotRequired[Any | None]
    response_json_schema: NotRequired[Any | None]
    response_mime_type: NotRequired[str | None]
    response_modalities: NotRequired[list[str] | None]
    routing_config: NotRequired[Any | None]
    safety_settings: NotRequired[list[Any] | None]
    service_tier: NotRequired[str | None]
    should_return_http_response: NotRequired[bool | None]
    speech_config: NotRequired[Any | None]
    thinking_config: NotRequired[Any | None]
    tool_config: NotRequired[Any | None]
