"""
Adapter for converting Agentle message objects to Cerebras message format.

This module provides the AgentleMessageToCerebrasMessageAdapter class, which transforms
Agentle's internal message representations (AssistantMessage, DeveloperMessage, UserMessage)
into the format expected by Cerebras's API (MessageAssistantMessageRequestTyped,
MessageSystemMessageRequestTyped, MessageUserMessageRequestTyped).

This adapter is an essential component of the Cerebras provider implementation,
enabling seamless communication between Agentle's unified interface and Cerebras's
specific API requirements.
"""

from cerebras.cloud.sdk.types.chat.completion_create_params import (
    MessageAssistantMessageRequestTyped,
    MessageSystemMessageRequestTyped,
    MessageUserMessageRequestTyped,
)

from agentle.generations.models.messages.assistant_message import AssistantMessage
from agentle.generations.models.messages.developer_message import DeveloperMessage
from agentle.generations.models.messages.user_message import UserMessage
from rsb.adapters.adapter import Adapter


class AgentleMessageToCerebrasMessageAdapter(
    Adapter[
        AssistantMessage | DeveloperMessage | UserMessage,
        MessageSystemMessageRequestTyped
        | MessageAssistantMessageRequestTyped
        | MessageUserMessageRequestTyped,
    ]
):
    """
    Adapter for converting Agentle message objects to Cerebras message format.

    This class transforms Agentle's message objects (AssistantMessage, DeveloperMessage,
    UserMessage) into the corresponding message types expected by the Cerebras API.
    The adapter maps:
    - AssistantMessage → MessageAssistantMessageRequestTyped with role="assistant"
    - DeveloperMessage → MessageSystemMessageRequestTyped with role="system"
    - UserMessage → MessageUserMessageRequestTyped with role="user"

    The adapter concatenates all text parts from the original message to form
    the content of the Cerebras message.
    """

    def adapt(
        self, _f: AssistantMessage | DeveloperMessage | UserMessage
    ) -> (
        MessageSystemMessageRequestTyped
        | MessageAssistantMessageRequestTyped
        | MessageUserMessageRequestTyped
    ):
        """
        Convert an Agentle message to its corresponding Cerebras message format.

        This method takes an Agentle message object and transforms it into the
        appropriate Cerebras message type based on the message class. It extracts
        and concatenates all text from the message parts to create the content.

        Args:
            _f: The Agentle message object to convert (AssistantMessage,
                DeveloperMessage, or UserMessage).

        Returns:
            The corresponding Cerebras message object (MessageAssistantMessageRequestTyped,
            MessageSystemMessageRequestTyped, or MessageUserMessageRequestTyped).
        """
        match _f:
            case AssistantMessage():
                return MessageAssistantMessageRequestTyped(
                    role="assistant", content="".join(str(p) for p in _f.parts)
                )
            case DeveloperMessage():
                return MessageSystemMessageRequestTyped(
                    role="system", content="".join(str(p) for p in _f.parts)
                )
            case UserMessage():
                return MessageUserMessageRequestTyped(
                    role="user", content="".join(str(p) for p in _f.parts)
                )
