from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class Hint(BaseModel):
    name: str | None = Field(default=None)
