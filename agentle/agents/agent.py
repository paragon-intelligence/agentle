from __future__ import annotations

from collections.abc import AsyncGenerator, Callable, Sequence
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Literal, cast
from uuid import UUID

from rsb.coroutines.run_sync import run_sync
from rsb.models.base_model import BaseModel
from rsb.models.config_dict import ConfigDict
from rsb.models.field import Field
from rsb.models.mimetype import MimeType

from agentle.agents.agent_config import AgentConfig
from agentle.agents.agent_output import AgentOutput
from agentle.agents.context import Context
from agentle.agents.models.agent_skill import AgentSkill
from agentle.agents.models.authentication import Authentication
from agentle.agents.models.capabilities import Capabilities
from agentle.agents.models.middleware.response_middleware import ResponseMiddleware

# from gat.agents.models.middleware.response_middleware import ResponseMiddleware
from agentle.agents.models.run_state import RunState

# from gat.agents.tools.agent_tool import AgentTool
from agentle.generations.models.generation.generation import Generation
from agentle.generations.models.message_parts.file import FilePart
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.message_parts.tool_execution_suggestion import (
    ToolExecutionSuggestion,
)
from agentle.generations.models.messages.assistant_message import AssistantMessage
from agentle.generations.models.messages.developer_message import DeveloperMessage
from agentle.generations.models.messages.message import Message
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.providers.base.generation_provider import (
    GenerationProvider,
)
from agentle.generations.tools.tool import Tool
from agentle.mcp.servers.mcp_server_protocol import MCPServerProtocol

type WithoutStructuredOutput = None
type _ToolName = str

type AgentInput = (
    str
    | Context
    | Sequence[AssistantMessage | DeveloperMessage | UserMessage]
    | UserMessage
    | TextPart
    | FilePart
    | Tool[Any]
    | Sequence[TextPart | FilePart | Tool[Any]]
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

    generation_provider: GenerationProvider = Field(...)
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

    defaultInputModes: Sequence[MimeType] = Field(
        default_factory=lambda: ["text/plain"]
    )
    """
    The set of interaction modes that the agent
    supports across all skills. This can be overridden per-skill.
    """

    defaultOutputModes: Sequence[MimeType] = Field(
        default_factory=lambda: ["text/plain", "application/json"]
    )
    """
    The set of interaction modes that the agent
    supports across all skills. This can be overridden per-skill.
    """

    skills: Sequence[AgentSkill] = Field(default_factory=list)
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

    tools: Sequence[Tool[Any] | Callable[..., object]] = Field(default_factory=list)
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

        state = RunState[T_Schema].init_state()

        # Populate the generations list with the generations with only the response middleware wrapped response attribute as a parsed attribute
        generations: list[Generation[T_Schema]] = []

        filtered_tools: list[Tool[Any]] = [
            Tool.from_callable(tool) if callable(tool) else tool for tool in self.tools
        ]

        while not state.task_completed and state.iteration < self.config.maxIterations:
            response_schema_type: (
                type[ResponseMiddleware[T_Schema]] | type[ResponseMiddleware[str]]
            ) = (
                ResponseMiddleware[str]
                if self.response_schema is None
                else ResponseMiddleware[T_Schema]
            )

            called_tools_names: set[str] = {
                tool_execution_suggestion.tool_name
                for tool_execution_suggestion in state.called_tools
            }

            filtered_tools = [
                tool for tool in filtered_tools if tool.name not in called_tools_names
            ]

            generation = await self.generation_provider.create_generation_async(
                model=self.model,
                messages=context.messages,
                response_schema=response_schema_type,
                generation_config=self.config.generationConfig,
                tools=filtered_tools,
            )

            parsed = cast(  # type: ignore[reportUnnecessaryCast]
                ResponseMiddleware[T_Schema] | ResponseMiddleware[str],
                generation.parsed,
            )

            generation_copy: Generation[T_Schema | None] = generation.clone(
                new_parseds=[
                    parsed.response
                    if self.response_schema is not None
                    and not isinstance(parsed.response, str)
                    else None
                ]
            )

            generations.append(cast(Generation[T_Schema], generation_copy))

            called_tools: dict[ToolExecutionSuggestion, Any] = {}
            available_tools: dict[_ToolName, Tool[Any]] = {
                tool.name: tool for tool in filtered_tools
            }
            for tool_execution_suggestion in generation.tool_calls:
                called_tools[tool_execution_suggestion] = available_tools[
                    tool_execution_suggestion.tool_name
                ].call(**tool_execution_suggestion.args)

            state.update(
                task_completed=parsed.task_completed,
                last_response=parsed.response,
                called_tools=called_tools,
                tool_calls_amount=generation.tool_calls_amount,
                iteration=state.iteration + 1,
            )

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

    def clone(
        self,
        *,
        new_name: str | None = None,
        new_instructions: str | None = None,
        new_tools: Sequence[Tool | Callable[..., object]] | None = None,
        new_config: AgentConfig | None = None,
        new_model: str | None = None,
        new_version: str | None = None,
        new_documentation_url: str | None = None,
        new_capabilities: Capabilities | None = None,
        new_authentication: Authentication | None = None,
        new_default_input_modes: Sequence[str] | None = None,
        new_default_output_modes: Sequence[str] | None = None,
        new_skills: Sequence[AgentSkill] | None = None,
        new_mcp_servers: Sequence[MCPServerProtocol] | None = None,
        new_generation_provider: GenerationProvider | None = None,
        new_url: str | None = None,
    ) -> Agent[T_Schema]:
        return Agent[T_Schema](
            name=new_name or self.name,
            instructions=new_instructions or self.instructions,
            tools=new_tools or self.tools,
            config=new_config or self.config,
            model=new_model or self.model,
            version=new_version or self.version,
            documentationUrl=new_documentation_url or self.documentationUrl,
            capabilities=new_capabilities or self.capabilities,
            authentication=new_authentication or self.authentication,
            defaultInputModes=new_default_input_modes or self.defaultInputModes,
            defaultOutputModes=new_default_output_modes or self.defaultOutputModes,
            skills=new_skills or self.skills,
            mcp_servers=new_mcp_servers or self.mcp_servers,
            generation_provider=new_generation_provider or self.generation_provider,
            url=new_url or self.url,
        )

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
        elif callable(input) and not isinstance(input, Tool):
            return Context(
                messages=[
                    DeveloperMessage(parts=[TextPart(text=instructions)]),
                    UserMessage(parts=[TextPart(text=input())]),
                ]
            )
        elif isinstance(input, UserMessage):
            # Tratar mensagem do usuário
            return Context(
                messages=[DeveloperMessage(parts=[TextPart(text=instructions)]), input]
            )
        elif (
            isinstance(input, TextPart)
            or isinstance(input, FilePart)
            or isinstance(input, Tool)
        ):
            # Tratar parte única
            input_parts = [input]
            return Context(
                messages=[
                    DeveloperMessage(parts=[TextPart(text=instructions)]),
                    UserMessage(parts=input_parts),
                ],
            )
        else:
            # Verificar se é uma sequência de mensagens ou partes
            if isinstance(input[0], AssistantMessage | DeveloperMessage | UserMessage):
                # Sequência de mensagens
                return Context(messages=list(cast(Sequence[Message], input)))
            elif isinstance(input[0], TextPart) or isinstance(input[0], FilePart):
                # Sequência de partes
                return Context(
                    messages=[
                        DeveloperMessage(parts=[TextPart(text=instructions)]),
                        UserMessage(
                            parts=list(
                                cast(
                                    Sequence[TextPart | FilePart | Tool],
                                    input,
                                )
                            )
                        ),
                    ]
                )

        # Retorno padrão para evitar erro de tipo
        return Context(messages=[DeveloperMessage(parts=[TextPart(text=instructions)])])
