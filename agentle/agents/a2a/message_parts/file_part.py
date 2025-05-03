"""
A2A File Message Part

This module defines the FilePart class, which represents a file component of a message
in the A2A protocol. File parts allow agents and users to exchange binary or text files,
enabling transfer of documents, images, and other file-based content.
"""

from __future__ import annotations


from typing import Literal

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.a2a.models.file import File


class FilePart(BaseModel):
    """
    Represents a file component of a message in the A2A protocol.

    FilePart objects contain file data that can be included in messages between
    users and agents. They are used for exchanging files such as documents, images,
    and other binary or text content.

    Attributes:
        type: The type of the message part, always "file"
        file: A File object containing the file data and metadata

    Example:
        ```python
        from agentle.agents.a2a.message_parts.file_part import FilePart
        from agentle.agents.a2a.models.file import File
        from agentle.agents.a2a.messages.message import Message
        import base64

        # Create a file with base64-encoded data
        image_data = base64.b64encode(open("image.png", "rb").read()).decode("utf-8")
        file_obj = File(
            name="image.png",
            mimeType="image/png",
            bytes=image_data
        )

        # Create a file part
        file_part = FilePart(file=file_obj)

        # Use it in a message
        message = Message(
            role="user",
            parts=[file_part]
        )

        # Alternatively, create a file part with a URI
        uri_file = File(
            name="document.pdf",
            mimeType="application/pdf",
            uri="https://example.com/documents/report.pdf"
        )
        uri_file_part = FilePart(file=uri_file)
        ```
    """

    type: Literal["file"] = Field(default="file")
    """The type of the message part, always "file" """

    file: File
    """A File object containing the file data and metadata"""

    @property
    def text(self) -> str:
        """
        Get the text content of the file.
        """
        return f"<file>{self.file.name}</file>"
