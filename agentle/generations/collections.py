"""
Message sequence manipulation module for Agentle generations.

This module provides the MessageSequence class, which represents an immutable
sequence of messages that can be manipulated through functional operations.
MessageSequence facilitates conversational context management by providing
methods to append, insert, and filter messages without modifying the original sequence.

MessageSequence is a critical component for managing conversation history in agents,
allowing flexible manipulation of message order and content while preserving
the original sequences. This is particularly useful when preparing messages
for submission to language models.

Example:
```python
from agentle.generations.collections.message_sequence import MessageSequence
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.models.messages.assistant_message import AssistantMessage
from agentle.generations.models.message_parts.text import TextPart

# Create a message sequence
messages = MessageSequence([
    UserMessage(parts=[TextPart(text="Hello, how are you?")]),
    AssistantMessage(parts=[TextPart(text="I'm doing well, thank you!")])
])

# Append a new message
updated_messages = messages.append([
    UserMessage(parts=[TextPart(text="What's the weather today?")])
])

# Insert a message before the last one
with_context = updated_messages.append_before_last_message(
    "Consider the user is in New York."
)

# Filter messages
user_messages = messages.filter(lambda msg: isinstance(msg, UserMessage))
```
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

from rsb.collections.readonly_collection import ReadonlyCollection

from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.messages.developer_message import DeveloperMessage
from agentle.generations.models.messages.message import Message
from agentle.generations.models.messages.user_message import UserMessage

logger = logging.getLogger(__name__)


class MessageSequence(ReadonlyCollection[Message]):
    """
    An immutable sequence of messages with functional manipulation methods.

    MessageSequence extends ReadonlyCollection to provide an immutable sequence
    of Message objects with methods for creating new sequences through operations
    like append, insert, and filter. It plays a key role in managing conversational
    contexts in the Agentle framework.

    The class follows functional programming principles, where operations return
    new MessageSequence instances rather than modifying the original sequence.
    This ensures thread safety and predictable behavior when managing message
    history.

    Example:
        ```python
        # Create a message sequence from existing messages
        sequence = MessageSequence([user_message, assistant_message])

        # Create a new sequence with an additional message
        new_sequence = sequence.append([new_message])

        # Insert a context message before the last message
        with_context = new_sequence.append_before_last_message(context_message)
        ```
    """

    def append(self, messages: Sequence[Message]) -> MessageSequence:
        """
        Creates a new sequence with additional messages appended.

        This method takes a sequence of messages and creates a new MessageSequence
        that includes all the original messages followed by the new messages.
        The original sequence remains unchanged.

        Args:
            messages: A sequence of Message objects to append to the current sequence.

        Returns:
            MessageSequence: A new message sequence containing all messages from
                the original sequence followed by the provided messages.

        Example:
            ```python
            # Start with an initial conversation
            sequence = MessageSequence([initial_message])

            # Add a follow-up message
            updated = sequence.append([follow_up_message])

            # Original sequence remains unchanged
            assert len(sequence.elements) == 1
            assert len(updated.elements) == 2
            ```
        """
        return MessageSequence(elements=list(self.elements) + list(messages))

    def append_before_last_message(self, message: Message | str) -> MessageSequence:
        """
        Appends a message before the last message in the sequence.

        This method is particularly useful for inserting contextual information
        or system instructions before the final user message, while maintaining
        the original message as the last one in the sequence.

        Args:
            message: The message to insert before the last message. Can be either
                a Message object or a string. If a string is provided, it will be
                converted to a UserMessage with a TextPart.

        Returns:
            MessageSequence: A new message sequence with the provided message
                inserted before the last message of the original sequence.

        Example:
            before: [A, B, C]
            after: [A, B, D, C]  (where D is the inserted message)

        Note:
            If the provided message is an empty string, the original sequence
            is returned unchanged.
        """
        if isinstance(message, str):
            if message.strip() == "":
                return self

            message = UserMessage(parts=[TextPart(text=message)])

        return MessageSequence(
            elements=list(self.elements[:-1]) + [message] + list(self.elements[-1:])
        )

    def filter(self, predicate: Callable[[Message], bool]) -> MessageSequence:
        """
        Creates a new sequence containing only messages that satisfy a predicate.

        This method allows filtering messages based on custom criteria defined
        by a callable predicate function. It returns a new MessageSequence with
        only the messages for which the predicate returns True.

        Args:
            predicate: A function that takes a Message and returns a boolean value.
                Only messages for which this function returns True will be included
                in the resulting sequence.

        Returns:
            MessageSequence: A new message sequence containing only the messages
                that satisfy the predicate.

        Example:
            ```python
            # Create a sequence with different message types
            sequence = MessageSequence([user_msg, system_msg, assistant_msg])

            # Filter to keep only user messages
            user_msgs = sequence.filter(lambda msg: isinstance(msg, UserMessage))

            # Filter based on content
            contains_question = sequence.filter(
                lambda msg: "?" in msg.parts[0].text if hasattr(msg.parts[0], "text") else False
            )
            ```
        """
        return MessageSequence(elements=list(filter(predicate, self.elements)))

    def without_developer_prompt(self) -> MessageSequence:
        """
        Creates a new sequence with all DeveloperMessage instances removed.

        This method is useful when preparing a message sequence for display
        or when developer messages (system prompts/instructions) should be
        excluded from the conversation being processed.

        Returns:
            MessageSequence: A new message sequence with all DeveloperMessage
                instances filtered out.

        Example:
            ```python
            # Create a sequence with developer instructions and user messages
            sequence = MessageSequence([
                DeveloperMessage(parts=[TextPart(text="You are a helpful assistant.")]),
                UserMessage(parts=[TextPart(text="Hello!")])
            ])

            # Remove the developer message
            user_facing = sequence.without_developer_prompt()

            # The result only contains the user message
            assert len(user_facing.elements) == 1
            assert isinstance(user_facing.elements[0], UserMessage)
            ```
        """
        return MessageSequence(
            list(
                filter(
                    lambda message: not isinstance(message, DeveloperMessage),
                    self.elements,
                )
            )
        )
