"""
Module for text-based message parts.
"""

from typing import Literal

from rsb.decorators.value_objects import valueobject
from rsb.models.base_model import BaseModel
from rsb.models.field import Field


@valueobject
class TextPart(BaseModel):
    """
    Represents a plain text part of a message.

    This class is used for textual content within messages in the system.
    """
    type: Literal["text"] = Field(
        default="text",
        description="Discriminator field to identify this as a text message part.",
    )

    text: str = Field(description="The textual content of the message part.")

