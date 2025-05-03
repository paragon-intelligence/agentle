from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from blacksheep.server.controllers import Controller
from rsb.adapters.adapter import Adapter
from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.agent import Agent
from agentle.agents.agent_input import AgentInput
from agentle.agents.agent_run_output import AgentRunOutput
from agentle.generations.models.message_parts.file import FilePart
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.messages.assistant_message import AssistantMessage
from agentle.generations.models.messages.developer_message import DeveloperMessage
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.tools.tool import Tool

try:
    import blacksheep
    from blacksheep.server.controllers import Controller
except ImportError:
    pass

if TYPE_CHECKING:
    from blacksheep.server.controllers import Controller


class _AgentRunCommand(BaseModel):
    input: (
        str
        | Sequence[AssistantMessage | DeveloperMessage | UserMessage]
        | Sequence[TextPart | FilePart | Tool[Any]]
        | TextPart
        | FilePart
    ) = Field(
        description="Input of the agent",
        examples=[
            "Hello, how are you?",
        ],
    )


class AgentToBlackSheepRouteHandler(Adapter[Agent[Any], "type[Controller]"]):
    def adapt(self, _f: Agent[Any]) -> type[Controller]:
        """
        Creates a BlackSheep router for the agent.
        """
        import blacksheep
        from blacksheep.server.openapi.common import (
            ContentInfo,
            ResponseInfo,
        )
        from blacksheep.server.openapi.v3 import OpenAPIHandler
        from openapidocs.v3 import Info

        agent = _f
        endpoint = (
            agent.endpoint
            or f"/api/v1/agents/{agent.name.lower().replace(' ', '_')}/run"
        )

        docs = OpenAPIHandler(
            info=Info(
                title=agent.name,
                version="1.0.0",
                summary=agent.description,
            )
        )

        agent_return_type = agent.run_async.__annotations__["return"]

        class _Run(Controller):
            @docs(
                responses={
                    200: ResponseInfo(
                        description="The agent run output",
                        content=[ContentInfo(type=AgentRunOutput[agent_return_type])],
                    )
                }
            )
            @blacksheep.post(endpoint)
            async def run(
                self, input: blacksheep.FromJSON[_AgentRunCommand]
            ) -> AgentRunOutput[Any]:
                async with agent.with_mcp_servers_async():
                    result = await agent.run_async(cast(AgentInput, input.value.input))
                    return result

        # Rename the class to match the agent name
        _Run.__name__ = f"{agent.name}Controller"
        return _Run
