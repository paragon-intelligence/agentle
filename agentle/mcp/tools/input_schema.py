"""
Input Schema definition for MCP tools.

This module provides the InputSchema class which defines the expected parameters
for tools in the Model Control Protocol (MCP) system. It follows a JSON Schema-like
structure to validate and document tool parameters.
"""

from typing import Literal

from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class InputSchema(BaseModel):
    """
    Schema definition for tool inputs in the MCP system.

    This class defines the expected parameters for a tool using a JSON Schema-like
    structure. It provides metadata about parameter types, defaults, requirements,
    and descriptions that can be used for validation and documentation.

    Attributes:
        properties (dict[str, object]): Dictionary mapping parameter names to their
            schema definitions, including type information, defaults, and descriptions
        type (Literal["object"]): The schema type, always set to "object" for MCP tools
    """

    properties: dict[str, object] = Field(
        description="Tool specific parameters and their schema definitions"
    )
    type: Literal["object"] = Field(
        default="object",
        description="Schema type, always 'object' for MCP tool input schemas",
    )
