from __future__ import annotations
from typing import TYPE_CHECKING
from rsb.adapters.adapter import Adapter

from agentle.generations.tools.tool import Tool

if TYPE_CHECKING:
    from cerebras.cloud.sdk.types.chat.completion_create_params import (
        Tool as CerebrasTool,
    )


class AgentleToolToCerebrasToolAdapter(Adapter[Tool, "CerebrasTool"]):
    def adapt(self, tool: Tool) -> CerebrasTool:
        from cerebras.cloud.sdk.types.chat.completion_create_params import (
            ToolTyped as CerebrasToolTyped,
        )
        from cerebras.cloud.sdk.types.chat.completion_create_params import (
            ToolFunctionTyped as CerebrasToolFunctionTyped,
        )

        return CerebrasToolTyped(
            function=CerebrasToolFunctionTyped(
                name=tool.name,
                description=tool.description,
                parameters=tool.parameters,
            ),
            type="tool",
        )
