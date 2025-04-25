from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class ResourceDescriptionWithTemplate(BaseModel):
    uriTemplate: str
    name: str
    description: str | None = Field(default=None)
    mimeType: str | None = Field(default=None)
