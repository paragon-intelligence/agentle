from __future__ import annotations

from typing import TYPE_CHECKING

from rsb.adapters.adapter import Adapter

from agentle.generations.models.messages.assistant_message import AssistantMessage
from agentle.generations.models.messages.developer_message import DeveloperMessage
from agentle.generations.models.messages.user_message import UserMessage

if TYPE_CHECKING:
    from openai.types.chat.chat_completion_assistant_message_param import (
        ChatCompletionAssistantMessageParam,
    )
    from openai.types.chat.chat_completion_developer_message_param import (
        ChatCompletionDeveloperMessageParam,
    )
    from openai.types.chat.chat_completion_user_message_param import (
        ChatCompletionUserMessageParam,
    )


class AgentleMessageToOpenaiMessageAdapter(
    Adapter[
        AssistantMessage | DeveloperMessage | UserMessage,
        "ChatCompletionDeveloperMessageParam | ChatCompletionUserMessageParam | ChatCompletionAssistantMessageParam",
    ]
):
    def adapt(
        self, _f: AssistantMessage | DeveloperMessage | UserMessage
    ) -> (
        ChatCompletionAssistantMessageParam
        | ChatCompletionDeveloperMessageParam
        | ChatCompletionUserMessageParam
    ):
        message = _f

        match message:
            case AssistantMessage():
                ...
            case DeveloperMessage():
                ...
            case UserMessage():
                ...

        raise NotImplementedError
