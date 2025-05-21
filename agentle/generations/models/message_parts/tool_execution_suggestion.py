"""
Module for tool execution suggestion message parts.
"""

import uuid
from typing import Literal

from rsb.models.base_model import BaseModel
from rsb.models.field import Field
from rsb.models.config_dict import ConfigDict


class ToolExecutionSuggestion(BaseModel):
    """
    Represents a suggestion to execute a specific tool.

    This class is used to model tool execution suggestions within messages,
    including the tool name and arguments.
    """

    type: Literal["tool_execution"] = Field(
        default="tool_execution",
        description="Discriminator field to identify this as a tool execution suggestion.",
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this tool execution suggestion.",
    )

    tool_name: str = Field(description="The name of the tool to be executed.")

    args: dict[str, object] = Field(
        default_factory=dict,
        description="The arguments to pass to the tool during execution.",
    )

    model_config = ConfigDict(frozen=True)

    @property
    def text(self) -> str:
        """
        Returns a text representation of the tool execution suggestion.

        Returns:
            str: A text representation containing the tool name and arguments.
        """
        return f"Tool: {self.tool_name}\nArgs: {self.args}"

    def __str__(self) -> str:
        return self.text
