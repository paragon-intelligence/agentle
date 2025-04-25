from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Sequence
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Literal, cast
from uuid import UUID

from rsb.coroutines.run_sync import run_sync
from rsb.models.base_model import BaseModel
from rsb.models.config_dict import ConfigDict
from rsb.models.field import Field

from agentle.agents.agent_config import AgentConfig
from agentle.agents.agent_output import AgentOutput
from agentle.agents.context import Context
from agentle.agents.models.agent_skill import AgentSkill
from agentle.agents.models.authentication import Authentication
from agentle.agents.models.capabilities import Capabilities

# from gat.agents.models.middleware.response_middleware import ResponseMiddleware
from agentle.agents.models.run_state import RunState
from agentle.agents.pipelines.agent_pipeline import AgenticPipeline
from agentle.agents.squads.agent_squad import AgentSquad

# from gat.agents.tools.agent_tool import AgentTool
from agentle.generations.models.generation.generation import Generation
from agentle.generations.models.message_parts.file import FilePart
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.message_parts.tool_declaration import ToolDeclaration
from agentle.generations.models.messages.assistant_message import AssistantMessage
from agentle.generations.models.messages.developer_message import DeveloperMessage
from agentle.generations.models.messages.message import Message
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.providers.base.generation_provider import (
    GenerationProvider,
)
from agentle.mcp.servers.mcp_server_protocol import MCPServerProtocol

type WithoutStructuredOutput = None

type AgentInput = (
    str
    | Context
    | Sequence[AssistantMessage | DeveloperMessage | UserMessage]
    | UserMessage
    | TextPart
    | FilePart
    | ToolDeclaration
    | Sequence[TextPart | FilePart | ToolDeclaration]
    | Callable[[], str]
)

if TYPE_CHECKING:
    from fastapi import APIRouter


class Agent[T_Schema = WithoutStructuredOutput](BaseModel):
    # Agent-to-agent protocol fields
    name: str
    """
    Human readable name of the agent.
    (e.g. "Recipe Agent")
    """

    description: str = Field(default="An AI agent")
    """
    A human-readable description of the agent. Used to assist users and
    other agents in understanding what the agent can do.
    (e.g. "Agent that helps users with recipes and cooking.")
    """

    url: str = Field(default="in-memory")
    """
    A URL to the address the agent is hosted at.
    """

    generation_provider: GenerationProvider
    """
    The service provider of the agent
    """

    version: str = Field(default="0.0.1")
    """
    The version of the agent - format is up to the provider. (e.g. "1.0.0")
    """

    documentationUrl: str | None = Field(default=None)
    """
    A URL to documentation for the agent.
    """

    capabilities: Capabilities = Field(default_factory=Capabilities)
    """
    Optional capabilities supported by the agent.
    """

    authentication: Authentication = Field(
        default_factory=lambda: Authentication(schemes=["basic"])
    )
    """
    Authentication requirements for the agent.
    Intended to match OpenAPI authentication structure.
    """

    defaultInputModes: Sequence[str] = Field(default_factory=lambda: ["text/plain"])
    """
    The set of interaction modes that the agent
    supports across all skills. This can be overridden per-skill.
    """

    defaultOutputModes: Sequence[str] = Field(
        default_factory=lambda: ["text/plain", "application/json"]
    )
    """
    The set of interaction modes that the agent
    supports across all skills. This can be overridden per-skill.
    """

    skills: Sequence[AgentSkill]
    """
    Skills are a unit of capability that an agent can perform.
    """

    # Library-specific fields
    model: str
    """
    The model to use for the agent's service provider.
    """

    instructions: str | Callable[[], str] | Sequence[str] = Field(
        default="You are a helpful assistant."
    )
    """
    The instructions to use for the agent.
    """

    response_schema: type[T_Schema] | None = None
    """
    The schema of the response to be returned by the agent.
    """

    mcp_servers: Sequence[MCPServerProtocol] = Field(default_factory=list)
    """
    The MCP servers to use for the agent.
    """

    tools: Sequence[Callable[..., object] | ToolDeclaration] = Field(
        default_factory=list
    )
    """
    The tools to use for the agent.
    """

    config: AgentConfig = Field(default_factory=AgentConfig)
    """
    The configuration for the agent.
    """

    # Internal fields
    model_config = ConfigDict(frozen=True)

    @property
    def uid(self) -> str:
        return str(hash(self))

    @asynccontextmanager
    async def with_mcp_servers(self) -> AsyncGenerator[None, None]:
        for server in self.mcp_servers:
            await server.connect()
        try:
            yield
        finally:
            for server in self.mcp_servers:
                await server.cleanup()

    def run(
        self,
        input: AgentInput,
        task_id: UUID | None = None,
        timeout: float | None = None,
    ) -> AgentOutput[T_Schema]:
        return run_sync(self.run_async, timeout=timeout, input=input, task_id=task_id)

    async def run_async(
        self,
        input: AgentInput,
        task_id: UUID | None = None,
    ) -> AgentOutput[T_Schema]:
        final_instructions: str = self._convert_instructions_to_str(self.instructions)
        context: Context = self._convert_input_to_context(
            input, instructions=final_instructions
        )

        # usar ResponseMiddleware[T_Schema] como response_schema. vai ajudar a saber se o agente terminou ou não de resolver o problema
        # enviado pelo usuario.

        state = RunState[str](
            task_completed=False,
            iteration=0,
            tool_calls_amount=0,
            called_tools={*()},
            last_response=None,
        )

        while not state.task_completed and state.iteration < self.config.maxIterations:
            state.iteration += 1
            # TODO(arthur): Implement the agent logic here.
            # Gerar uma resposta ate a tarefa ser classificada como concluida pelo agente.

        # TODO(arthur): Implement the agent logic here

        return AgentOutput[T_Schema](
            generation=Generation[T_Schema].mock(), final_context=context
        )

    def to_http_router(
        self, path: str, type: Literal["fastapi"] = "fastapi"
    ) -> APIRouter:
        match type:
            case "fastapi":
                return self._to_fastapi_router(path=path)

    def _to_fastapi_router(self, path: str) -> APIRouter:
        from fastapi import APIRouter

        router = APIRouter()
        # TODO(arthur): create the endpoint here.

        router.add_api_route(path=path, endpoint=self.run)
        return router

    def _convert_instructions_to_str(
        self, instructions: str | Callable[[], str] | Sequence[str]
    ) -> str:
        """
        Convert the instructions to an AgentInstructions object.
        """
        if isinstance(instructions, str):
            return instructions
        elif callable(instructions):
            return instructions()
        else:
            return "".join(instructions)

    def _convert_input_to_context(
        self,
        input: AgentInput,
        instructions: str,
    ) -> Context:
        if isinstance(input, str):
            return Context(
                messages=[
                    DeveloperMessage(parts=[TextPart(text=instructions)]),
                    UserMessage(parts=[TextPart(text=input)]),
                ],
            )
        elif isinstance(input, Context):
            return input
        elif callable(input):
            return Context(
                messages=[
                    DeveloperMessage(parts=[TextPart(text=instructions)]),
                    UserMessage(parts=[TextPart(text=input())]),
                ]
            )
        elif isinstance(input, UserMessage) or (
            isinstance(input, AssistantMessage | DeveloperMessage | UserMessage)
            and not isinstance(input, DeveloperMessage)
        ):
            # Tratar mensagem do usuário
            return Context(
                messages=[DeveloperMessage(parts=[TextPart(text=instructions)]), input]
            )
        elif (
            isinstance(input, TextPart)
            or isinstance(input, FilePart)
            or isinstance(input, ToolDeclaration)
        ):
            # Tratar parte única
            return Context(
                messages=[
                    DeveloperMessage(parts=[TextPart(text=instructions)]),
                    UserMessage(parts=[input]),
                ]
            )
        else:
            # Verificar se é uma sequência de mensagens ou partes
            if isinstance(input[0], AssistantMessage | DeveloperMessage | UserMessage):
                # Sequência de mensagens
                return Context(messages=list(cast(Sequence[Message], input)))
            elif (
                isinstance(input[0], TextPart)
                or isinstance(input[0], FilePart)
                or isinstance(input[0], ToolDeclaration)
            ):
                # Sequência de partes
                return Context(
                    messages=[
                        DeveloperMessage(parts=[TextPart(text=instructions)]),
                        UserMessage(
                            parts=list(
                                cast(
                                    Sequence[TextPart | FilePart | ToolDeclaration],
                                    input,
                                )
                            )
                        ),
                    ]
                )

        # Retorno padrão para evitar erro de tipo
        return Context(messages=[DeveloperMessage(parts=[TextPart(text=instructions)])])

    def __add__(self, other: Agent[Any]) -> AgentSquad: ...

    def __or__(self, other: Agent) -> AgenticPipeline: ...
