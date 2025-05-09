"""
Adapter module for converting Agentle Message objects to Google AI Content format.

This module provides the MessageToGoogleContentAdapter class, which transforms Agentle's
internal Message representations into the Content format expected by Google's
Generative AI APIs. This conversion is necessary when sending messages to Google's
AI models as part of a conversation or prompt.

The adapter handles role mapping between Agentle's role system and Google's role system,
and delegates part conversion to the PartToGooglePartAdapter. It ensures that messages
being sent to Google's models are properly formatted according to their API requirements.

This adapter is typically used internally by the GoogleGenerationProvider when
preparing message content to be sent to Google's API.

Example:
```python
from agentle.generations.providers.google._adapters.message_to_google_content_adapter import (
    MessageToGoogleContentAdapter
)
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.models.message_parts.text import TextPart

# Create an Agentle message
user_message = UserMessage(parts=[TextPart(text="Hello, world!")])

# Create an adapter
adapter = MessageToGoogleContentAdapter()

# Convert to Google's format
google_content = adapter.adapt(user_message)

# Now use with Google's API
response = model.generate_content([google_content])
```
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from rsb.adapters.adapter import Adapter

from agentle.generations.models.messages.message import Message
from agentle.generations.providers.google.adapters.part_to_google_part_adapter import (
    PartToGooglePartAdapter,
)
from agentle.generations.tools.tool import Tool

if TYPE_CHECKING:
    from google.genai.types import Content


class MessageToGoogleContentAdapter(Adapter[Message, "Content"]):
    """
    Adapter for converting Agentle Message objects to Google AI Content format.

    This adapter transforms Agentle's internal Message objects (UserMessage,
    AssistantMessage, DeveloperMessage) into the Content format expected by
    Google's Generative AI APIs. It handles role mapping between Agentle's role
    system and Google's role system, and delegates the conversion of message
    parts to the PartToGooglePartAdapter.

    The adapter is a key component of Agentle's provider abstraction layer,
    allowing the framework to use a unified internal message representation while
    still being able to communicate with Google's specific API formats.

    Role mapping:
    - "user" → "user"
    - "assistant" → "model"
    - "developer" → "model"

    Attributes:
        part_adapter (PartToGooglePartAdapter): Adapter used to convert individual
            message parts from Agentle's format to Google's format.

    Example:
        ```python
        # Convert a user message
        from agentle.generations.models.messages.user_message import UserMessage
        from agentle.generations.models.message_parts.text import TextPart

        user_message = UserMessage(parts=[
            TextPart(text="What's the weather like in London?")
        ])

        adapter = MessageToGoogleContentAdapter()
        google_content = adapter.adapt(user_message)

        # Convert an assistant message
        from agentle.generations.models.messages.assistant_message import AssistantMessage

        assistant_message = AssistantMessage(parts=[
            TextPart(text="The weather in London is currently sunny.")
        ])

        google_assistant_content = adapter.adapt(assistant_message)
        ```
    """

    part_adapter: PartToGooglePartAdapter

    def __init__(self, part_adapter: PartToGooglePartAdapter | None = None) -> None:
        """
        Initialize the adapter with an optional part adapter.

        Args:
            part_adapter: Optional adapter for converting message parts. If not
                provided, a new PartToGooglePartAdapter is created.
        """
        super().__init__()
        self.part_adapter = part_adapter or PartToGooglePartAdapter()

    @override
    def adapt(self, _f: Message) -> "Content":
        """
        Convert an Agentle Message to a Google AI Content object.

        This method transforms an Agentle Message into a Google Content object
        by mapping the message role and converting each message part. Tool parts
        are filtered out as they cannot be declared by an AI in Google's system.

        Args:
            _f: The Agentle Message object to convert. This can be any Message
                subclass such as UserMessage, AssistantMessage, or DeveloperMessage.

        Returns:
            Content: A Google AI Content object containing the converted message
                with appropriate role and parts.

        Example:
            ```python
            # Converting a message with multiple parts
            from agentle.generations.models.messages.user_message import UserMessage
            from agentle.generations.models.message_parts.text import TextPart
            from agentle.generations.models.message_parts.file import FilePart

            # User message with text and image
            user_message = UserMessage(parts=[
                TextPart(text="What's in this image?"),
                FilePart(data=image_bytes, mime_type="image/jpeg")
            ])

            # Convert to Google format
            google_content = adapter.adapt(user_message)

            # Role is mapped to "user" and both parts are converted
            assert google_content.role == "user"
            assert len(google_content.parts) == 2
            ```
        """
        from google.genai.types import Content

        part_adapter = self.part_adapter or PartToGooglePartAdapter()

        match _f.role:
            case "assistant":
                role = "model"
            case "developer":
                role = "model"
            case "user":
                role = "user"

        return Content(
            parts=[
                part_adapter.adapt(part)
                for part in _f.parts
                if not isinstance(  # Impossible that an AI declares a tool
                    part, Tool
                )
            ],
            role=role,
        )
