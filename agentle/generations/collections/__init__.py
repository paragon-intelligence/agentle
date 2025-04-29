"""
Collections for manipulating message sequences in the Agentle framework.

This module provides collection classes that help manage and manipulate sequences
of messages for AI generation. The collections maintain immutability and follow
functional programming principles, where operations return new instances rather
than modifying the original collections.

The primary collection provided is MessageSequence, which represents an immutable
sequence of messages with methods for appending, inserting, and filtering messages
without altering the original sequence. This is particularly useful for managing
conversation history and context in AI agents and applications.

These collections serve as building blocks for implementing complex conversational
flows, maintaining context windows, and preparing structured inputs for language models.
"""

from .message_sequence import MessageSequence

__all__: list[str] = ["MessageSequence"]
