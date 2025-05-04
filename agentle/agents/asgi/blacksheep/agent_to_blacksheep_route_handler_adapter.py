from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from blacksheep.server.controllers import Controller
from rsb.adapters.adapter import Adapter
from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.a2a.a2a_interface import A2AInterface
from agentle.agents.a2a.tasks.task import Task
from agentle.agents.a2a.tasks.task_get_result import TaskGetResult
from agentle.agents.a2a.tasks.task_send_params import TaskSendParams
from agentle.agents.a2a.tasks.task_query_params import TaskQueryParams
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


class _TaskSendRequest(BaseModel):
    task_params: TaskSendParams = Field(
        description="Parameters for sending a task",
    )


class _TaskQueryRequest(BaseModel):
    query_params: TaskQueryParams = Field(
        description="Parameters for querying a task",
    )


class _TaskCancelRequest(BaseModel):
    task_id: str = Field(
        description="ID of the task to cancel",
    )


class AgentToBlackSheepRouteHandlerAdapter(Adapter[Agent[Any], "type[Controller]"]):
    def adapt(self, _f: Agent[Any] | A2AInterface[Any]) -> type[Controller]:
        """
        Creates a BlackSheep router for the agent.
        """
        match _f:
            case Agent():
                return self._adapt_agent(_f)
            case _:
                return self._adapt_a2a_interface(_f)

    def _adapt_a2a_interface(self, _f: A2AInterface[Any]) -> type[Controller]:
        """
        Creates a BlackSheep controller for the A2A interface.
        """
        import blacksheep

        a2a = _f

        class A2AController(Controller):
            @blacksheep.post("/api/v1/tasks/send")
            async def send_task(
                self, input: blacksheep.FromJSON[_TaskSendRequest]
            ) -> Task:
                """
                Send a task to the agent
                """
                return await a2a.task_manager.send(
                    task_params=input.value.task_params, agent=a2a.agent
                )

            @blacksheep.post("/api/v1/tasks/get")
            async def get_task(
                self, input: blacksheep.FromJSON[_TaskQueryRequest]
            ) -> TaskGetResult:
                """
                Get task results
                """
                return await a2a.task_manager.get(
                    query_params=input.value.query_params, agent=a2a.agent
                )

            @blacksheep.post("/api/v1/tasks/cancel")
            async def cancel_task(
                self, input: blacksheep.FromJSON[_TaskCancelRequest]
            ) -> bool:
                """
                Cancel a task
                """
                return await a2a.task_manager.cancel(task_id=input.value.task_id)

            @blacksheep.ws("/api/v1/notifications")
            async def subscribe_notifications(self, websocket: Any) -> None:
                """
                Subscribe to push notifications via WebSocket
                """
                #TODO(arthur): Implement this
                try:
                    # Keep the connection alive and handle incoming messages
                    while True:
                        await websocket.receive_text()
                        # Process incoming messages if needed
                except Exception:
                    # Connection closed or error occurred
                    pass

        return A2AController

    def _adapt_agent(self, _f: Agent[Any]) -> type[Controller]:
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
            ) -> AgentRunOutput[dict[str, Any]]:
                async with agent.with_mcp_servers_async():
                    result = await agent.run_async(cast(AgentInput, input.value.input))
                    return result

        # Rename the class to match the agent name
        _Run.__name__ = f"{agent.name}Controller"
        return _Run
