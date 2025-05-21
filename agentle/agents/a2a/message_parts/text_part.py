"""
A2A Text Message Part

This module defines the TextPart class, which represents a text component of a message
in the A2A protocol. Text parts are the most common message components, containing
natural language content exchanged between users and agents.
"""

from __future__ import annotations

from typing import Literal

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.prompts.models.prompt import Prompt


class TextPart(BaseModel):
    """
    Represents a text component of a message in the A2A protocol.

    TextPart objects contain text content that can be included in messages between
    users and agents. They are the most common type of message part and are used
    for natural language communication.

    Attributes:
        type: The type of the message part, always "text"
        text: The text content of the message part

    Example:
        ```python
        from agentle.agents.a2a.message_parts.text_part import TextPart
        from agentle.agents.a2a.messages.message import Message

        # Create a text part
        text_part = TextPart(text="Hello, how can I help you today?")

        # Use it in a message
        message = Message(
            role="agent",
            parts=[text_part]
        )

        # Access the text content
        print(message.parts[0].text)  # "Hello, how can I help you today?"
        ```
    """

    type: Literal["text"] = Field(default="text")
    """The type of the message part, always "text" """

    text: str | Prompt
    """The text content of the message part"""
