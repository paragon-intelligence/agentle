"""
Sampling Request Module

This module defines the SamplingRequest class used to structure requests
for language model completions or generations.
"""

from typing import Literal, Sequence

from agentle.generations.models.messages.user_message import UserMessage
from agentle.mcp.sampling.messages.assistant_message import AssistantMessage
from agentle.mcp.sampling.model_preferences import ModelPreference
from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class SamplingRequest(BaseModel):
    """
    Represents a request for a language model completion or generation.

    This class encapsulates all the parameters needed to request a completion
    from a language model, including conversation history, model preferences,
    system instructions, and sampling parameters.

    Attributes:
        messages: Sequence of messages representing the conversation history.
        modelPreferences: Optional preferences for model selection and behavior.
        systemPrompt: Optional system-level instructions for the model.
        includeContext: Optional setting for context inclusion scope.
        temperature: Optional temperature setting for controlling randomness.
        maxTokens: Maximum number of tokens to generate.
        stopSequences: Optional sequences that will stop generation when encountered.
        metadata: Optional additional metadata for the request.
    """

    messages: Sequence[AssistantMessage | UserMessage] = Field(
        description="Sequence of messages representing the conversation history",
        examples=[
            [
                {
                    "role": "user",
                    "content": {
                        "type": "text",
                        "text": "Hello, how can you help me today?",
                    },
                },
                {
                    "role": "assistant",
                    "content": {
                        "type": "text",
                        "text": "I can answer questions and assist with tasks.",
                    },
                },
            ]
        ],
    )
    modelPreferences: ModelPreference | None = Field(
        default=None,
        description="Optional preferences for model selection and behavior",
    )
    systemPrompt: str | None = Field(
        default=None,
        description="Optional system-level instructions for the model",
        examples=[
            "You are a helpful AI assistant that answers questions accurately and concisely."
        ],
    )
    includeContext: Literal["none", "thisServer", "allServers"] | None = Field(
        default=None,
        description="Optional setting specifying the scope of context to include: 'none' for no context, 'thisServer' for context from the current server only, or 'allServers' for context from all available servers",
    )
    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional temperature setting for controlling randomness in generation (0.0 for deterministic, 1.0 for maximum randomness)",
        examples=[0.7],
    )
    maxTokens: float = Field(
        description="Maximum number of tokens to generate in the completion",
        examples=[1024],
    )
    stopSequences: Sequence[str] | None = Field(
        default=None,
        description="Optional sequences that will stop generation when encountered",
        examples=[["###", "END"]],
    )
    metadata: dict[str, object] = Field(
        default_factory=dict,
        description="Optional additional metadata for the request",
        examples=[{"user_id": "u123", "session_id": "s456"}],
    )
