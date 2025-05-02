"""
A2A Message to Generation Message Adapter

This module provides an adapter for converting A2A messages to Generation messages.
The adapter ensures compatibility between the A2A protocol's message format and the
generation system's message format.
"""

from typing import Literal

from rsb.adapters.adapter import Adapter

from agentle.agents.a2a.message_parts.adapters.agent_part_to_generation_part_adapter import (
    AgentPartToGenerationPartAdapter,
)
from agentle.agents.a2a.messages.message import Message
from agentle.generations.models.messages.assistant_message import AssistantMessage
from agentle.generations.models.messages.user_message import UserMessage


class MessageToGenerationMessageAdapter(
    Adapter[Message, UserMessage | AssistantMessage]
):
    """
    Adapter for converting A2A messages to Generation messages.

    This adapter transforms Message objects from the A2A protocol into
    UserMessage and AssistantMessage objects used in the generation system.
    It handles the mapping of roles and conversion of message parts.

    Example:
        ```python
        from agentle.agents.a2a.messages.message import Message
        from agentle.agents.a2a.message_parts.text_part import TextPart
        from agentle.agents.a2a.messages.message_to_generation_message_adapter import MessageToGenerationMessageAdapter

        # Create an A2A message
        a2a_message = Message(
            role="user",
            parts=[TextPart(text="Hello, world!")]
        )

        # Convert to Generation message
        adapter = MessageToGenerationMessageAdapter()
        gen_message = adapter.adapt(a2a_message)

        print(gen_message.role)  # "user"
        print(gen_message.parts[0].text)  # "Hello, world!"
        ```
    """

    def adapt(self, _f: Message) -> UserMessage | AssistantMessage:
        """
        Adapts an A2A message to a Generation message.

        This method converts a Message from the A2A protocol into a UserMessage
        or AssistantMessage used in the generation system. It maps the roles and
        converts each message part using the AgentPartToGenerationPartAdapter.

        Args:
            _f: The A2A message to adapt

        Returns:
            The adapted Generation message (UserMessage or AssistantMessage)

        Example:
            ```python
            adapter = MessageToGenerationMessageAdapter()
            gen_message = adapter.adapt(a2a_message)
            ```
        """
        roles: dict[Literal["user", "agent"], Literal["user", "assistant"]] = {
            "user": "user",
            "agent": "assistant",
        }

        part_adapter = AgentPartToGenerationPartAdapter()

        # Convert parts
        parts = [part_adapter.adapt(part) for part in _f.parts]

        # Create appropriate message type based on role
        role = roles[_f.role]
        if role == "user":
            return UserMessage(parts=parts)
        return AssistantMessage(parts=parts)
