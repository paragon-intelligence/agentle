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
    def adapt(
        self, _f: AssistantMessage | DeveloperMessage | UserMessage
    ) -> (
        MessageSystemMessageRequestTyped
        | MessageAssistantMessageRequestTyped
        | MessageUserMessageRequestTyped
    ):
        match _f:
            case AssistantMessage():
                return MessageAssistantMessageRequestTyped(
                    role="assistant", content="".join(p.text for p in _f.parts)
                )
            case DeveloperMessage():
                return MessageSystemMessageRequestTyped(
                    role="system", content="".join(p.text for p in _f.parts)
                )
            case UserMessage():
                return MessageUserMessageRequestTyped(
                    role="user", content="".join(p.text for p in _f.parts)
                )
