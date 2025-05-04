from rsb.models.base_model import BaseModel
from rsb.models.field import Field
from rsb.models.any_url import AnyUrl
from typing import Literal


class UrlKnowledge(BaseModel):
    """
    A URL to the address the agent is hosted at.
    """

    type: Literal["url"] = Field(default="url")
    url: AnyUrl
