"""
Configuration module for Agentle agents.

This module defines the configuration options that control the behavior of agents in the Agentle framework.
The primary class, AgentConfig, encapsulates settings related to generation parameters,
tool call limits, and iteration bounds that govern how agents operate.

Example:
```python
from agentle.agents.agent_config import AgentConfig
from agentle.generations.models.generation.generation_config import GenerationConfig

# Create a custom configuration with specific settings
config = AgentConfig(
    generationConfig=GenerationConfig(temperature=0.7, top_p=0.95),
    maxToolCalls=20,
    maxIterations=15
)

# Use the configuration when creating an agent
agent = Agent(
    # ... other parameters ...
    config=config
)
```
"""

from typing import NotRequired, TypedDict
from agentle.generations.models.generation.generation_config import GenerationConfig
from agentle.generations.models.generation.generation_config_dict import (
    GenerationConfigDict,
)


class AgentConfigDict(TypedDict):
    """
    Configuration class for Agentle agents.

    This class defines the configurable parameters that control how an agent behaves,
    particularly with respect to generation settings and limitations on tool usage.

    Attributes:
        generationConfig (GenerationConfig): Configuration for the underlying language model generation.
            Controls parameters like temperature, top_p, max_tokens, etc.
        maxToolCalls (int): Maximum number of individual tool calls the agent can make during execution.
            Limits the total number of external tool invocations to prevent excessive resource usage.
        maxIterations (int): Maximum number of back-and-forth iterations the agent can perform.
            Controls how many rounds of reasoning/tool usage the agent can go through before completing.

    Example:
        ```python
        # Default configuration
        default_config = AgentConfig()

        # Custom configuration with higher limits
        custom_config = AgentConfig(
            generationConfig=GenerationConfig(temperature=0.8),
            maxToolCalls=20,
            maxIterations=15
        )
        ```

    Note:
        The configuration enforces that generationConfig.n = 1, as multiple
        generations per request are not supported for agents.
    """

    generationConfig: NotRequired[GenerationConfig | GenerationConfigDict]
    """Configuration for the language model generation process."""

    maxToolCalls: NotRequired[int]
    """Maximum number of tool calls allowed during agent execution."""

    maxIterations: NotRequired[int]
    """Maximum number of agent reasoning iterations before terminating."""
