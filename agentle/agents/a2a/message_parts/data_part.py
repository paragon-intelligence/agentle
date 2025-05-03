"""
A2A Data Message Part

This module defines the DataPart class, which represents a structured data component
of a message in the A2A protocol. Data parts allow agents and users to exchange
structured information such as JSON objects, enabling richer interactions beyond
plain text.
"""

from typing import Any, Literal

from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class DataPart(BaseModel):
    """
    Represents a structured data component of a message in the A2A protocol.

    DataPart objects contain structured data (as a dictionary) that can be included
    in messages between users and agents. They are used for exchanging structured
    information such as JSON objects, enabling richer interactions beyond plain text.

    Attributes:
        type: The type of the message part, always "data"
        data: A dictionary containing the structured data

    Example:
        ```python
        from agentle.agents.a2a.message_parts.data_part import DataPart
        from agentle.agents.a2a.messages.message import Message

        # Create a data part with structured information
        weather_data = DataPart(data={
            "location": "San Francisco",
            "temperature": 68,
            "condition": "Partly Cloudy",
            "forecast": [
                {"day": "Monday", "high": 70, "low": 55},
                {"day": "Tuesday", "high": 72, "low": 56}
            ]
        })

        # Use it in a message
        message = Message(
            role="agent",
            parts=[weather_data]
        )

        # Access the structured data
        temperature = message.parts[0].data["temperature"]  # 68
        ```
    """

    type: Literal["data"] = Field(default="data")
    """The type of the message part, always "data" """

    data: dict[str, Any]
    """A dictionary containing the structured data"""

    @property
    def text(self) -> str:
        """
        Get the text content of the data part.
        """
        return str(self.data)
