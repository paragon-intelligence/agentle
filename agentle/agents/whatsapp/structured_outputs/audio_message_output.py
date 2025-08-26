from typing import Literal

from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class AudioMessageOutput(BaseModel):
    type: Literal["audio"] = Field(default="audio")
    audio: str = Field(
        description="A string of what will be said in the audio in english."
    )
    voice: Literal["male", "female"] = Field(default="female")
