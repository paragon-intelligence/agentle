"""
Prompt module for MCP.

This module defines the Prompt class which represents a named prompt template
in the Model Control Protocol (MCP) system. Prompts are structured templates
that can be filled with arguments and used for generating text with language models.
"""

from collections.abc import Sequence

from agentle.mcp.prompts.prompt_argument import PromptArgument
from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class Prompt(BaseModel):
    """
    Definition of a prompt template in the MCP system.

    Prompt represents a named template that can be used for generating text with
    language models. It includes metadata such as name, description, and a list
    of expected arguments, which helps document and validate prompt usage.

    Attributes:
        name (str): The unique name of the prompt
        description (str | None): A human-readable description of the prompt,
            explaining its purpose and usage
        arguments (Sequence[PromptArgument]): A list of arguments that the prompt
            expects or accepts
    """

    name: str = Field(..., description="Unique name of the prompt")
    description: str | None = Field(
        default=None, description="Human-readable description of the prompt's purpose"
    )
    arguments: Sequence[PromptArgument] = Field(
        ..., description="List of arguments that the prompt expects or accepts"
    )
