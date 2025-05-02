"""
Generation Message to A2A Message Adapter

This module provides an adapter for converting between generation messages and A2A messages.
The adapter ensures compatibility between the generation system's message format and the
standardized A2A message format.
"""

from typing import Literal

from rsb.adapters.adapter import Adapter

from agentle.agents.a2a.message_parts.adapters.generation_part_to_agent_part_adapter import (
    GenerationPartToAgentPartAdapter,
)
from agentle.agents.a2a.messages.message import Message
from agentle.generations.models.messages.assistant_message import AssistantMessage
from agentle.generations.models.messages.user_message import UserMessage


class GenerationMessageToMessageAdapter(
    Adapter[UserMessage | AssistantMessage, Message]
):
    """
    Adapter for converting generation messages to A2A messages.

    This adapter transforms UserMessage and AssistantMessage objects from the generation
    system into standardized Message objects used in the A2A protocol. It handles the
    mapping of roles and conversion of message parts.

    Example:
        ```python
        from agentle.generations.models.messages.user_message import UserMessage
        from agentle.generations.models.message_parts.text import TextPart as GenTextPart
        from agentle.agents.a2a.messages.generation_message_to_message_adapter import GenerationMessageToMessageAdapter

        # Create a generation system message
        gen_message = UserMessage(
            parts=[GenTextPart(text="Hello, world!")]
        )

        # Convert to A2A message
        adapter = GenerationMessageToMessageAdapter()
        a2a_message = adapter.adapt(gen_message)

        print(a2a_message.role)  # "user"
        print(a2a_message.parts[0].text)  # "Hello, world!"
        ```
    """

    def adapt(self, _f: UserMessage | AssistantMessage) -> Message:
        """
        Adapts a generation message to an A2A message.

        This method converts a UserMessage or AssistantMessage from the generation system
        into a standardized Message used in the A2A protocol. It maps the roles and
        converts each message part using the GenerationPartToAgentPartAdapter.

        Args:
            _f: The generation message to adapt (UserMessage or AssistantMessage)

        Returns:
            Message: The adapted A2A message

        Example:
            ```python
            adapter = GenerationMessageToMessageAdapter()
            a2a_message = adapter.adapt(generation_message)
            ```
        """
        roles: dict[Literal["user", "assistant"], Literal["user", "agent"]] = {
            "user": "user",
            "assistant": "agent",
        }

        part_adapter = GenerationPartToAgentPartAdapter()
        return Message(
            role=roles[_f.role],
            parts=[part_adapter.adapt(part) for part in _f.parts],
        )
