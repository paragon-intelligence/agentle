"""
A2A Message Model

This module defines the Message class, which represents a communication unit in the A2A protocol.
Messages are exchanged between users and agents, containing one or more parts (text, data, files).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.a2a.message_parts.data_part import DataPart
from agentle.agents.a2a.message_parts.file_part import FilePart
from agentle.agents.a2a.message_parts.text_part import TextPart


class Message(BaseModel):
    """
    Represents a message exchanged between a user and an agent.

    A message consists of one or more parts, which can be text, files, or structured data.
    Each message has a role indicating whether it's from a user or an agent.

    Attributes:
        role: Role of the message sender, either "user" or "agent"
        parts: Sequence of message parts (text, files, or data)
        metadata: Optional metadata associated with the message

    Example:
        ```python
        from agentle.agents.a2a.messages.message import Message
        from agentle.agents.a2a.message_parts.text_part import TextPart
        from agentle.agents.a2a.message_parts.data_part import DataPart

        # Create a simple text message
        text_message = Message(
            role="user",
            parts=[TextPart(text="What is the weather like today?")]
        )

        # Create a message with both text and structured data
        data_message = Message(
            role="agent",
            parts=[
                TextPart(text="Here's the current weather:"),
                DataPart(data={"temperature": 72, "condition": "sunny"})
            ],
            metadata={"source": "weather_api", "timestamp": "2023-06-15T12:00:00Z"}
        )
        ```
    """

    role: Literal["user", "agent"]
    """Role of the message sender, either "user" or "agent" """

    parts: Sequence[TextPart | FilePart | DataPart]
    """Sequence of message parts (text, files, or data)"""

    metadata: dict[str, Any] | None = Field(default=None)
    """Optional metadata associated with the message"""
