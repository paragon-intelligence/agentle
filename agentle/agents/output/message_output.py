from typing import Literal

from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class MessageOutput(BaseModel):
    type: Literal["message"] = Field(default="message")
