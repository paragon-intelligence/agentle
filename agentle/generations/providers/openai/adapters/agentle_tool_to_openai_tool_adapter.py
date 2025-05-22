from __future__ import annotations

from typing import TYPE_CHECKING

from rsb.adapters.adapter import Adapter

from agentle.generations.tools.tool import Tool

if TYPE_CHECKING:
    from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam


class AgentleToolToOpenaiToolAdapter(Adapter[Tool, "ChatCompletionToolParam"]):
    def adapt(self, tool: Tool) -> ChatCompletionToolParam:
        from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
        from openai.types.shared_params.function_definition import FunctionDefinition

        return ChatCompletionToolParam(
            function=FunctionDefinition(
                name=tool.name,
                description=tool.description or "",
                parameters=tool.parameters,
            ),
            type="function",
        )
