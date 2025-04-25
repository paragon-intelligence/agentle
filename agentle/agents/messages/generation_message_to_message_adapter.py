from typing import Literal

from rsb.adapters.adapter import Adapter

from agentle.agents.message_parts.generation_part_to_agent_part_adapter import (
    GenerationPartToAgentPartAdapter,
)
from agentle.agents.messages.message import Message
from agentle.generations.models.messages.assistant_message import AssistantMessage
from agentle.generations.models.messages.user_message import UserMessage


class GenerationMessageToMessageAdapter(
    Adapter[UserMessage | AssistantMessage, Message]
):
    def adapt(self, _f: UserMessage | AssistantMessage) -> Message:
        roles: dict[Literal["user", "assistant"], Literal["user", "agent"]] = {
            "user": "user",
            "assistant": "agent",
        }

        part_adapter = GenerationPartToAgentPartAdapter()
        return Message(
            role=roles[_f.role],
            parts=[part_adapter.adapt(part) for part in _f.parts],
        )
