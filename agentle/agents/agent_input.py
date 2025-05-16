from collections.abc import Callable, Sequence
from io import BytesIO, StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agentle.agents.context import Context
from agentle.generations.models.message_parts.file import FilePart
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.messages.developer_message import DeveloperMessage
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.tools.tool import Tool
from agentle.mcp.sampling.messages.assistant_message import AssistantMessage
from agentle.prompts.models.prompt import Prompt
from agentle.parsing.parsed_document import ParsedDocument

if TYPE_CHECKING:
    import datetime

    import numpy as np
    import pandas as pd
    from PIL import Image
    from pydantic import BaseModel as PydanticBaseModel

type AgentInput = (
    str
    | Prompt
    | Context
    | Sequence[AssistantMessage | DeveloperMessage | UserMessage]
    | UserMessage
    | TextPart
    | FilePart
    | Tool[Any]
    | Sequence[TextPart | FilePart | Tool[Any]]
    | Callable[[], str]
    | pd.DataFrame
    | np.ndarray[Any, Any]
    | Image.Image
    | bytes
    | dict[str, Any]
    | list[Any]
    | tuple[Any, ...]
    | set[Any]
    | frozenset[Any]
    | datetime.datetime
    | datetime.date
    | datetime.time
    | Path
    | BytesIO
    | StringIO
    | PydanticBaseModel
    | ParsedDocument
)
