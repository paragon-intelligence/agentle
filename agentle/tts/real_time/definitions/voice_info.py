from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.stt.real_time.definitions.language_code import LanguageCode
from agentle.tts.real_time.definitions.voice_gender import VoiceGender


class VoiceInfo(BaseModel):
    """Information about a voice."""

    voice_id: str
    name: str
    language: LanguageCode
    gender: VoiceGender
    accent: str | None = Field(default=None, description="Regional accent")
    age_group: str | None = Field(default=None, description="Young, adult, elderly")
    description: str | None = Field(default=None)
    preview_url: str | None = Field(default=None)
