import mimetypes
from typing import Literal

from rsb.decorators.value_objects import valueobject
from rsb.models.base_model import BaseModel
from rsb.models.field import Field


@valueobject
class FilePart(BaseModel):
    data: bytes
    mime_type: str
    type: Literal["file"] = Field(default="file")

    @property
    def text(self) -> str:
        return f"<file>\n{self.mime_type}\n </file>"

    def __post_init__(self) -> None:
        allowed_mimes = mimetypes.types_map.values()
        mime_type_unknown = self.mime_type not in allowed_mimes
        if mime_type_unknown:
            raise ValueError(
                f"The provided MIME ({self.mime_type}) is not in the list of official mime types: {allowed_mimes}"
            )
