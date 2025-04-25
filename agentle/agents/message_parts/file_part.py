from __future__ import annotations


from typing import Literal

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.models.file import File


class FilePart(BaseModel):
    type: Literal["file"] = Field(default="file")
    file: File
