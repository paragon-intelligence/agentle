from __future__ import annotations

import base64
from typing import TYPE_CHECKING, override

from rsb.adapters.adapter import Adapter

from agentle.generations.models.message_parts.file import FilePart
from agentle.generations.models.message_parts.text import TextPart

if TYPE_CHECKING:
    from openai.types.chat.chat_completion_content_part_image_param import (
        ChatCompletionContentPartImageParam,
    )
    from openai.types.chat.chat_completion_content_part_input_audio_param import (
        ChatCompletionContentPartInputAudioParam,
    )
    from openai.types.chat.chat_completion_content_part_text_param import (
        ChatCompletionContentPartTextParam,
    )


class AgentlePartToOpenaiPartAdapter(
    Adapter[
        TextPart | FilePart,
        "ChatCompletionContentPartImageParam | ChatCompletionContentPartInputAudioParam | ChatCompletionContentPartTextParam",
    ]
):
    @override
    def adapt(
        self, _f: TextPart | FilePart
    ) -> (
        ChatCompletionContentPartTextParam
        | ChatCompletionContentPartImageParam
        | ChatCompletionContentPartInputAudioParam
    ):
        part = _f

        match part:
            case TextPart():
                return ChatCompletionContentPartTextParam(text=str(part), type="text")
            case FilePart():
                mime_type = part.mime_type
                if mime_type.startswith("image/"):
                    return ChatCompletionContentPartImageParam(
                        image_url={
                            "url": base64.b64encode(part.data).decode(),
                            "detail": "auto",
                        },
                        type="image_url",
                    )
                elif mime_type.startswith("audio/"):
                    return ChatCompletionContentPartInputAudioParam(
                        input_audio={
                            "data": base64.b64encode(part.data).decode(),
                            "format": "mp3",
                        },
                        type="input_audio",
                    )
                else:
                    raise ValueError(f"Unsupported file type: {mime_type}")
