"""
Message Part Adapters

This package contains adapters for converting between different message part formats.
The adapters enable seamless integration between the A2A protocol and the generation
system by transforming message parts from one format to another.
"""

from agentle.agents.a2a.message_parts.adapters.generation_part_to_agent_part_adapter import (
    GenerationPartToAgentPartAdapter,
)
from agentle.agents.a2a.message_parts.adapters.agent_part_to_generation_part_adapter import (
    AgentPartToGenerationPartAdapter,
)

__all__ = ["GenerationPartToAgentPartAdapter", "AgentPartToGenerationPartAdapter"]
