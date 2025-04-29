"""
Prompt Argument module for MCP.

This module defines the PromptArgument class which represents a parameter
for prompts in the Model Control Protocol (MCP) system. Prompt arguments
provide structured metadata about the parameters that a prompt expects,
including name, description, and whether they are required.
"""

from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class PromptArgument(BaseModel):
    """
    Definition of an argument for an MCP prompt.

    PromptArgument represents a parameter that can be passed to a prompt template.
    It includes metadata such as name, description, and whether the argument is
    required, which helps validate prompt usage and provide clear documentation.

    Attributes:
        name (str): The name of the argument, used as the key when providing values
        description (str | None): A human-readable description of the argument,
            explaining its purpose and expected values
        required (bool | None): Flag indicating whether this argument must be
            provided when using the prompt
    """

    name: str = Field(
        ..., description="Name of the argument, used as the key when providing values"
    )
    description: str | None = Field(
        default=None, description="Human-readable description of the argument's purpose"
    )
    required: bool | None = Field(
        default=None,
        description="Whether this argument must be provided when using the prompt",
    )
