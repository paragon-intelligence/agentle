"""
A2A File Model

This module defines the File class, which represents a file that can be exchanged between
agents and users in the A2A protocol. Files can contain binary data encoded as a string or
can be referenced by a URI.
"""

from __future__ import annotations
from typing import Self

from rsb.models.base_model import BaseModel
from rsb.models.field import Field
from rsb.models.model_validator import model_validator


class File(BaseModel):
    """
    Represents a file that can be exchanged between agents and users.

    File objects contain either binary data encoded as a string or a URI reference
    to a file. They include optional metadata such as name and MIME type.

    Attributes:
        name: The name of the file (optional)
        mimeType: The MIME type of the file (optional)
        bytes: The file content encoded as a string (e.g., base64)
        uri: A URI reference to the file

    Note:
        Exactly one of `bytes` or `uri` must be provided.

    Example:
        ```python
        from agentle.agents.a2a.models.file import File
        import base64

        # Create a file with base64-encoded content
        with open("document.pdf", "rb") as f:
            file_bytes = base64.b64encode(f.read()).decode("utf-8")

        file = File(
            name="document.pdf",
            mimeType="application/pdf",
            bytes=file_bytes
        )

        # Create a file with a URI reference
        uri_file = File(
            name="image.jpg",
            mimeType="image/jpeg",
            uri="https://example.com/images/photo.jpg"
        )
        ```
    """

    name: str | None = Field(default=None)
    """The name of the file (optional)"""

    mimeType: str | None = Field(default=None)
    """The MIME type of the file (optional)"""

    # OneOf
    bytes: str | None = Field(default=None)
    """The file content encoded as a string (e.g., base64)"""

    uri: str | None = Field(default=None)
    """A URI reference to the file"""

    @model_validator(mode="after")
    def check_one_of(self) -> Self:
        """
        Validates that exactly one of `bytes` or `uri` is provided.

        This validator ensures that a File object has either binary content
        or a URI reference, but not both or neither.

        Returns:
            Self: The validated File object

        Raises:
            ValueError: If neither bytes nor uri is provided

        Example:
            This validation is applied automatically when creating File objects:
            ```python
            # This will raise a ValueError
            invalid_file = File(name="test.txt")

            # This is valid
            valid_file = File(name="test.txt", bytes="SGVsbG8gd29ybGQ=")
            ```
        """
        if self.bytes is None and self.uri is None:
            raise ValueError("One of bytes or uri must be provided")
        return self
