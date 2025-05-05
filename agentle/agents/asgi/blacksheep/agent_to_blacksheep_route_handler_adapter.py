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
from agentle.agents.a2a.tasks.task_query_params import TaskQueryParams
from agentle.agents.a2a.tasks.task_send_params import TaskSendParams
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
        description="Input for the agent. Can be a simple string, a message part, or a sequence of messages or parts.",
        examples=[
            "Hello, how are you?",
            {"text": "What is the capital of France?", "type": "text"},
            [{"role": "user", "content": "Can you explain how neural networks work?"}],
        ],
    )


class _TaskSendRequest(BaseModel):
    task_params: TaskSendParams = Field(
        description="Parameters for sending a task to an agent. This includes the message to send and session information.",
        examples=[
            {
                "task_params": {
                    "message": {
                        "role": "user",
                        "parts": [
                            {
                                "type": "text",
                                "text": "How do I create a simple web server in Python?",
                            }
                        ],
                    },
                    "sessionId": "session-123",
                    "historyLength": 10,
                }
            },
            {
                "task_params": {
                    "message": {
                        "role": "user",
                        "parts": [
                            {
                                "type": "text",
                                "text": "What are the main features of Python 3.10?",
                            }
                        ],
                    },
                    "sessionId": "python-session",
                    "metadata": {"priority": "high", "category": "programming"},
                }
            },
        ],
    )


class _TaskQueryRequest(BaseModel):
    query_params: TaskQueryParams = Field(
        description="Parameters for querying a task. Specify the task ID to retrieve results and optionally limit the history length.",
        examples=[
            {"query_params": {"id": "task-123", "historyLength": 5}},
            {"query_params": {"id": "a1b2c3d4-5678-90ab-cdef-ghijklmnopqr"}},
        ],
    )


class _TaskCancelRequest(BaseModel):
    task_id: str = Field(
        description="ID of the task to cancel. This should be the UUID returned when the task was created.",
        examples=["task-123", "a1b2c3d4-5678-90ab-cdef-ghijklmnopqr"],
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
        from blacksheep.server.openapi.common import ContentInfo, ResponseInfo
        from blacksheep.server.openapi.v3 import OpenAPIHandler
        from openapidocs.v3 import Info

        a2a = _f

        # Create OpenAPI docs handler for the A2A interface
        docs = OpenAPIHandler(
            info=Info(
                title=f"{getattr(a2a.agent, 'name', 'Agent')} A2A Interface",
                version="1.0.0",
                summary=getattr(a2a.agent, "description", "Agent with A2A Interface"),
            )
        )

        class A2A(Controller):
            @docs(  # type: ignore
                responses={
                    200: ResponseInfo(
                        description="Task created successfully",
                        content=[ContentInfo(type=Task)],
                    ),
                    400: ResponseInfo(description="Invalid request parameters"),
                }
            )
            @blacksheep.post("/api/v1/tasks/send")
            async def send_task(
                self, input: blacksheep.FromJSON[_TaskSendRequest]
            ) -> Task:
                """
                Send a task to the agent asynchronously

                This endpoint allows you to submit a new task to the agent. The task will be processed
                asynchronously, and you can check its status later using the task ID that is returned.

                The request should include the message you want to send to the agent, along with any
                session information and optional parameters like history length and metadata.
                """
                return await a2a.task_manager.send(
                    task_params=input.value.task_params, agent=a2a.agent
                )

            @docs(  # type: ignore
                responses={
                    200: ResponseInfo(
                        description="Task retrieved successfully",
                        content=[ContentInfo(type=TaskGetResult)],
                    ),
                    404: ResponseInfo(description="Task not found"),
                }
            )
            @blacksheep.post("/api/v1/tasks/get")
            async def get_task(
                self, input: blacksheep.FromJSON[_TaskQueryRequest]
            ) -> TaskGetResult:
                """
                Get task results

                This endpoint allows you to retrieve the results of a previously submitted task.
                You need to provide the task ID that was returned when you created the task.

                Optionally, you can specify how much of the conversation history you want to include
                in the response using the historyLength parameter.
                """
                return await a2a.task_manager.get(
                    query_params=input.value.query_params, agent=a2a.agent
                )

            @docs(  # type: ignore
                responses={
                    200: ResponseInfo(
                        description="Task cancellation result",
                        content=[ContentInfo(type=bool)],
                    ),
                    404: ResponseInfo(description="Task not found"),
                }
            )
            @blacksheep.post("/api/v1/tasks/cancel")
            async def cancel_task(
                self, input: blacksheep.FromJSON[_TaskCancelRequest]
            ) -> bool:
                """
                Cancel a running task

                This endpoint allows you to cancel a task that is currently in progress.
                You need to provide the task ID that was returned when you created the task.

                Returns true if the task was successfully cancelled, false otherwise.
                """
                return await a2a.task_manager.cancel(task_id=input.value.task_id)

            @blacksheep.ws("/api/v1/notifications")
            async def subscribe_notifications(self, websocket: Any) -> None:
                """
                Subscribe to push notifications via WebSocket

                This endpoint allows you to subscribe to real-time notifications about task status changes.
                Connect to this endpoint using a WebSocket client to receive updates as they happen.

                The server will send JSON messages containing task updates whenever the status of a task changes.
                """
                # TODO(arthur): Implement this
                try:
                    # Keep the connection alive and handle incoming messages
                    while True:
                        await websocket.receive_text()
                        # Process incoming messages if needed
                except Exception:
                    # Connection closed or error occurred
                    pass

        return A2A

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

        class _Run(Controller):
            @docs(  # type: ignore
                responses={
                    200: ResponseInfo(
                        description="The agent run output",
                        content=[
                            ContentInfo(type=AgentRunOutput[Any | dict[str, Any]])
                        ],
                    )
                }
            )
            @blacksheep.post(endpoint)
            async def run(
                self, input: blacksheep.FromJSON[_AgentRunCommand]
            ) -> AgentRunOutput[dict[str, Any]]:
                """
                Run the agent with the provided input

                This endpoint allows you to send input to the agent and get a response synchronously.
                The agent will process your input and return the result immediately.

                You can provide input as a simple string, a structured message, or a sequence of messages.
                """
                async with agent.with_mcp_servers_async():
                    result = await agent.run_async(cast(AgentInput, input.value.input))
                    return result

        # Rename the class to match the agent name
        _Run.__name__ = f"{agent.name}Controller"
        return _Run
