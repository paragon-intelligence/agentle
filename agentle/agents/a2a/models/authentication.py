from collections.abc import Sequence
from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class Authentication(BaseModel):
    schemes: Sequence[str]
    """
    e.g. Basic, Bearer
    """

    credentials: str | None = Field(default=None)
    """
    Credentials a client should use for private cards
    """
