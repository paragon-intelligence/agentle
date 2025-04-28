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
    def append(self, messages: Sequence[Message]) -> MessageSequence:
        return MessageSequence(elements=list(self.elements) + list(messages))

    def append_before_last_message(self, message: Message | str) -> MessageSequence:
        """Appends a message before the last message. Keeps the last message.

        Args:
            message (Message): The message to append.

        Example:
            before: [A, B, C]
            after: [A, D, B, C]

        Returns:
            MessageSequence: The new message sequence.
        """
        if isinstance(message, str):
            if message.strip() == "":
                logger.warning("Message is empty. Skipping append.")
                return self

            message = UserMessage(parts=[TextPart(text=message)])

        return MessageSequence(
            elements=list(self.elements[:-1]) + [message] + list(self.elements[-1:])
        )

    def filter(self, predicate: Callable[[Message], bool]) -> MessageSequence:
        return MessageSequence(elements=list(filter(predicate, self.elements)))

    def without_developer_prompt(self) -> MessageSequence:
        return MessageSequence(
            list(
                filter(
                    lambda message: not isinstance(message, DeveloperMessage),
                    self.elements,
                )
            )
        )
