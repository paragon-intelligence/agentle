from typing import Literal

from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class Subtitles(BaseModel):
    format: Literal["srt"] = Field(
        default="srt",
        description="The format of the subtitles.",
    )

    subtitles: str = Field(
        default="",
        description="The subtitles.",
    )
