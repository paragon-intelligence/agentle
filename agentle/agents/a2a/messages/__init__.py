"""
A2A Messages Package

This package provides the message models and adapters used in the A2A protocol.
Messages represent the communication units exchanged between users and agents,
containing one or more parts (text, data, files).

The package includes:
- Message models defining the structure of user and agent messages
- Adapters for converting between different message formats
- Utility functions for message handling

Example:
    ```python
    from agentle.agents.a2a.messages.message import Message
    from agentle.agents.a2a.message_parts.text_part import TextPart

    # Create a simple message
    message = Message(
        role="user",
        parts=[TextPart(text="Hello, how can you help me today?")]
    )
    ```
"""

from agentle.agents.a2a.messages.message import Message
from agentle.agents.a2a.messages.generation_message_to_message_adapter import (
    GenerationMessageToMessageAdapter,
)

__all__ = ["Message", "GenerationMessageToMessageAdapter"]
