"""
Adapter components for the Cerebras AI provider integration in the Agentle framework.

This module contains adapter classes that transform requests to and responses from
Cerebras's AI API into standardized formats used throughout the Agentle framework.
These adapters serve as part of Agentle's provider abstraction layer, allowing the
framework to present a unified interface regardless of the underlying AI provider.

The adapters in this package handle various conversion tasks, including:
- Converting Agentle message objects to Cerebras message format
- Transforming Cerebras responses to Agentle Generation objects
- Supporting structured output parsing for type-safe responses

These adapters ensure that all provider-specific details of Cerebras's request and
response formats are processed and normalized to Agentle's internal representation,
maintaining consistency across different AI providers within the framework.
"""

from .agentle_message_to_cerebras_message_adapter import (
    AgentleMessageToCerebrasMessageAdapter,
)
from .cerebras_message_to_generated_assistant_message_adapter import (
    CerebrasMessageToGeneratedAssistantMessageAdapter,
)
from .completion_to_generation_adapter import CerebrasCompletionToGenerationAdapter

__all__ = [
    "AgentleMessageToCerebrasMessageAdapter",
    "CerebrasMessageToGeneratedAssistantMessageAdapter",
    "CerebrasCompletionToGenerationAdapter",
]
