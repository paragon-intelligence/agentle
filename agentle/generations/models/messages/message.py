from typing import Annotated

from agentle.generations.models.messages.assistant_message import AssistantMessage
from agentle.generations.models.messages.developer_message import DeveloperMessage
from agentle.generations.models.messages.user_message import UserMessage
from rsb.models.field import Field

type Message = Annotated[
    AssistantMessage | DeveloperMessage | UserMessage, Field(discriminator="role")
]
