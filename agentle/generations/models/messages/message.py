"""
Module defining the Message union type for all message types in the system.

This module provides a unified Message type that can represent any concrete message type
(AssistantMessage, DeveloperMessage, or UserMessage) using Annotated typing with a discriminator.
"""

from typing import Annotated

from agentle.generations.models.messages.assistant_message import AssistantMessage
from agentle.generations.models.messages.developer_message import DeveloperMessage
from agentle.generations.models.messages.user_message import UserMessage
from rsb.models.field import Field

# Message is a union type that can be any of the message types
# The Field with discriminator indicates that the "role" field is used
# to determine which concrete class to use when deserializing
type Message = Annotated[
    AssistantMessage | DeveloperMessage | UserMessage, Field(discriminator="role")
]
