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

from __future__ import annotations

from typing import Self

from rsb.models.base_model import BaseModel
from rsb.models.field import Field
from rsb.models.model_validator import model_validator

from agentle.generations.models.generation.generation_config import GenerationConfig
from agentle.generations.models.generation.generation_config_dict import (
    GenerationConfigDict,
)


class AgentConfig(BaseModel):
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

    generationConfig: GenerationConfig | GenerationConfigDict = Field(
        default_factory=GenerationConfig
    )
    """Configuration for the language model generation process."""

    maxToolCalls: int = Field(default=15)
    """Maximum number of tool calls allowed during agent execution."""

    maxIterations: int = Field(default=10)
    """Maximum number of agent reasoning iterations before terminating."""

    def clone(
        self,
        new_generation_config: GenerationConfig | GenerationConfigDict | None = None,
        new_max_tool_calls: int | None = None,
        new_max_iterations: int | None = None,
    ) -> AgentConfig:
        return AgentConfig(
            generationConfig=new_generation_config or self.generationConfig,
            maxToolCalls=new_max_tool_calls or self.maxToolCalls,
            maxIterations=new_max_iterations or self.maxIterations,
        )

    @property
    def generation_config(self) -> GenerationConfig:
        if isinstance(self.generationConfig, dict):
            return GenerationConfig.model_validate(self.generationConfig)
        return self.generationConfig

    @model_validator(mode="after")
    def validate_max_tool_calls(self) -> Self:
        """
        Validates that the generation configuration is compatible with agent requirements.

        This validator ensures that the number of generations (n) is set to 1, as agents
        do not (thanks GOD) support multiple parallel generations.

        Returns:
            Self: The validated AgentConfig instance.

        Raises:
            ValueError: If generationConfig.n is greater than 1.
        """
        if self.generation_config.n > 1:
            raise ValueError(
                "a number of choices > 1 is not supported for agents. This is NOT planned to be supported."
                + "If you want multiple choices/responses, just call the agent n times."
            )
        return self
