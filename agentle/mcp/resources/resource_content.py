from rsb.models.base64str import Base64Str
from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class ResourceContent(BaseModel):
    uri: str
    mimeType: str | None = Field(default=None)
    text: str | None = Field(default=None)
    blob: Base64Str | None = Field(default=None)
