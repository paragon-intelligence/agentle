"""
Collections module for the Agentle generations package.

This module provides utilities and data structures for working with collections of messages
and other generation-related objects. The collections package implements immutable data structures
that follow functional programming principles, where operations return new instances rather than
modifying the original objects.

Key components:

- MessageSequence: An immutable sequence of messages that can be manipulated through
  functional operations like append, insert, and filter without modifying the original sequence.
  This is particularly useful for managing conversation history in agents and preparing
  message sequences for submission to language models.

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

# Append a new message (returns a new sequence)
updated_messages = messages.append([
    UserMessage(parts=[TextPart(text="What's the weather today?")])
])

# Filter messages (returns a new sequence)
user_messages = messages.filter(lambda msg: isinstance(msg, UserMessage))
```
"""

# Import public modules and classes
from agentle.generations.collections.message_sequence import MessageSequence

__all__ = ["MessageSequence"]
