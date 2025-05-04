from rsb.models.base_model import BaseModel
from rsb.models.field import Field
from typing import Literal
from rsb.models.config_dict import ConfigDict


class TextPageItem(BaseModel):
    type: Literal["text"] = Field(default="text")

    text: str = Field(
        description="Value of the text item",
    )

    model_config = ConfigDict(frozen=True)
