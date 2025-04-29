from __future__ import annotations
from typing import Self

from rsb.models.base_model import BaseModel
from rsb.models.field import Field
from rsb.models.model_validator import model_validator


class File(BaseModel):
    name: str | None = Field(default=None)
    mimeType: str | None = Field(default=None)

    # OneOf
    bytes: str | None = Field(default=None)
    uri: str | None = Field(default=None)

    @model_validator(mode="after")
    def check_one_of(self) -> Self:
        if self.bytes is None and self.uri is None:
            raise ValueError("One of bytes or uri must be provided")
        return self
