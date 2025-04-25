from typing import Literal, Sequence

from agentle.generations.models.messages.user_message import UserMessage
from agentle.mcp.sampling.messages.assistant_message import AssistantMessage
from agentle.mcp.sampling.model_preferences import ModelPreference
from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class SamplingRequest(BaseModel):
    messages: Sequence[AssistantMessage | UserMessage] = Field(discriminator="role")
    modelPreferences: ModelPreference | None = Field(default=None)
    systemPrompt: str | None = Field(default=None)
    includeContext: Literal["none", "thisServer", "allServers"] | None = Field(
        default=None
    )
    temperature: float | None = Field(default=None, ge=0.0, le=1.0)
    maxTokens: float
    stopSequences: Sequence[str] | None = Field(default=None)
    metadata: dict[str, object] = Field(default_factory=dict)
