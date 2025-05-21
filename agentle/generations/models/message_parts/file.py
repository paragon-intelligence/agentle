"""
Module for file-based message parts.
"""

import mimetypes
from typing import Literal

from rsb.decorators.value_objects import valueobject
from rsb.models.base_model import BaseModel
from rsb.models.field import Field


@valueobject
class FilePart(BaseModel):
    """
    Represents a file attachment part of a message.

    This class handles binary file data with appropriate MIME type validation.
    """

    type: Literal["file"] = Field(
        default="file",
        description="Discriminator field to identify this as a file message part.",
    )

    data: bytes = Field(description="The binary content of the file.")

    mime_type: str = Field(
        description="The MIME type of the file, must be a valid MIME type from Python's mimetypes module."
    )

    @property
    def text(self) -> str:
        """
        Returns a text representation of the file part.

        Returns:
            str: A text representation containing the MIME type.
        """
        return f"<file>\n{self.mime_type}\n </file>"

    def __str__(self) -> str:
        return self.text

    def __post_init__(self) -> None:
        """
        Validates that the provided MIME type is official.

        Raises:
            ValueError: If the MIME type is not in the list of official MIME types.
        """
        allowed_mimes = mimetypes.types_map.values()
        mime_type_unknown = self.mime_type not in allowed_mimes
        if mime_type_unknown:
            raise ValueError(
                f"The provided MIME ({self.mime_type}) is not in the list of official mime types: {allowed_mimes}"
            )
