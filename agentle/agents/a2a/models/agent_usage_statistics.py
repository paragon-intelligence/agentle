"""
A2A Agent Usage Statistics Model

This module defines the AgentUsageStatistics class, which tracks token usage
statistics for agent operations in the A2A protocol. It provides a way to
monitor resource consumption during agent interactions.
"""

from rsb.models.base_model import BaseModel

from agentle.generations.models.generation.usage import Usage


class AgentUsageStatistics(BaseModel):
    """
    Tracks token usage statistics for agent operations.

    This class captures information about token usage during agent interactions,
    providing a way to monitor resource consumption, estimate costs, and track
    usage patterns.

    Attributes:
        token_usage: Usage statistics including input and output token counts

    Example:
        ```python
        from agentle.agents.a2a.models.agent_usage_statistics import AgentUsageStatistics
        from agentle.generations.models.generation.usage import Usage

        # Create usage statistics
        usage = Usage(
            input_tokens=150,
            output_tokens=300
        )

        stats = AgentUsageStatistics(token_usage=usage)

        # Access usage information
        print(f"Input tokens: {stats.token_usage.input_tokens}")
        print(f"Output tokens: {stats.token_usage.output_tokens}")
        print(f"Total tokens: {stats.token_usage.input_tokens + stats.token_usage.output_tokens}")
        ```
    """

    token_usage: Usage
    """Token usage statistics including input and output token counts"""
