from __future__ import annotations

import datetime
import json
from collections.abc import AsyncGenerator, Callable, MutableMapping, Sequence
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, Literal, cast

from mcp.types import Tool as MCPTool
from rsb.coroutines.run_sync import run_sync
from rsb.models.base_model import BaseModel
from rsb.models.config_dict import ConfigDict
from rsb.models.field import Field
from rsb.models.mimetype import MimeType

from agentle.agents.agent_config import AgentConfig
from agentle.agents.agent_run_output import AgentRunOutput
from agentle.agents.context import Context
from agentle.agents.errors.max_tool_calls_exceeded_error import (
    MaxToolCallsExceededError,
)
from agentle.agents.models.agent_skill import AgentSkill
from agentle.agents.models.agent_usage_statistics import AgentUsageStatistics
from agentle.agents.models.artifact import Artifact
from agentle.agents.models.authentication import Authentication
from agentle.agents.models.capabilities import Capabilities
from agentle.agents.models.run_state import RunState
from agentle.agents.tasks.task_state import TaskState
from agentle.generations.collections.message_sequence import MessageSequence
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

if TYPE_CHECKING:
    from io import BytesIO, StringIO
    from pathlib import Path

    import numpy as np
    import pandas as pd
    from fastapi import APIRouter
    from PIL import Image
    from pydantic import BaseModel as PydanticBaseModel

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
    | pd.DataFrame
    | np.ndarray[Any, Any]
    | Image.Image
    | bytes
    | dict[str, Any]
    | list[Any]
    | tuple[Any, ...]
    | set[Any]
    | frozenset[Any]
    | datetime.datetime
    | datetime.date
    | datetime.time
    | Path
    | BytesIO
    | StringIO
    | PydanticBaseModel
)


class Agent[T_Schema = WithoutStructuredOutput](BaseModel):
    # Agent-to-agent protocol fields
    name: str = Field(default="Agent")
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
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @property
    def uid(self) -> str:
        return str(hash(self))

    def has_tools(self) -> bool:
        return len(self.tools) > 0

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
        timeout: float | None = None,
    ) -> AgentRunOutput[T_Schema]:
        return run_sync(self.run_async, timeout=timeout, input=input)

    async def run_async(
        self,
        input: AgentInput,
    ) -> AgentRunOutput[T_Schema]:
        context: Context = self._convert_input_to_context(
            input, instructions=self._convert_instructions_to_str(self.instructions)
        )

        mcp_tools: list[MCPTool] = []
        for server in self.mcp_servers:
            tools = await server.list_tools()
            mcp_tools.extend(tools)

        agent_has_tools = self.has_tools() or len(mcp_tools) > 0
        if not agent_has_tools:
            generation: Generation[
                T_Schema
            ] = await self.generation_provider.create_generation_async(
                model=self.model,
                messages=context.messages,
                response_schema=self.response_schema,
                generation_config=self.config.generationConfig,
            )

            return AgentRunOutput[T_Schema](
                artifacts=[
                    Artifact(
                        name="Artifact",
                        description="End result of the task",
                        parts=list(
                            part
                            for part in generation.parts
                            if isinstance(part, TextPart)
                        ),
                        metadata=None,
                        index=0,
                        append=False,
                        last_chunk=None,
                    )
                ],
                task_status=TaskState.COMPLETED,
                usage=AgentUsageStatistics(token_usage=generation.usage),
                final_context=context,
                parsed=generation.parsed,
            )

        # Agent has tools. We must iterate until generate the final answer.

        all_tools: list[Tool[Any]] = [
            Tool.from_mcp_tool(tool) for tool in mcp_tools
        ] + [
            Tool.from_callable(tool) if callable(tool) else tool for tool in self.tools
        ]

        available_tools: MutableMapping[str, Tool[Any]] = {
            tool.name: tool for tool in all_tools
        }

        state = RunState[T_Schema].init_state()
        # Convert all tools in the array to Tool objects
        called_tools: dict[ToolExecutionSuggestion, Any] = {}
        while state.iteration < self.config.maxIterations:
            # Filter out tools that have already been called
            filtered_tools = [
                tool
                for tool in all_tools
                if tool.name
                not in {
                    tool_execution_suggestion.tool_name
                    for tool_execution_suggestion in called_tools
                }
            ]

            called_tools_prompt: str = (
                (
                    """<info>
                    The following are the other tool calls made by the agent:
                    </info>"""
                    + "\n"
                    + "\n".join(
                        [
                            f"""<tool_execution>
                    <tool_name>{tool_execution_suggestion.tool_name}</tool_name>
                    <args>{tool_execution_suggestion.args}</args>
                    <result>{result}</result>
                </tool_execution>"""
                            for tool_execution_suggestion, result in called_tools.items()
                        ]
                    )
                )
                if called_tools
                else ""
            )

            tool_call_generation = (
                await self.generation_provider.create_generation_async(
                    model=self.model,
                    messages=MessageSequence(context.messages)
                    .append_before_last_message(called_tools_prompt)
                    .elements,
                    generation_config=self.config.generationConfig,
                    tools=filtered_tools,
                )
            )

            agent_didnt_call_any_tool = tool_call_generation.tool_calls_amount() == 0
            if agent_didnt_call_any_tool:
                # Agent didn't call any tool. We can return/process the final answer.

                generation = await self.generation_provider.create_generation_async(
                    model=self.model,
                    messages=context.messages,
                    response_schema=self.response_schema,
                    generation_config=self.config.generationConfig,
                )

                return self._build_agent_run_output(
                    artifact_name="Artifact",
                    artifact_description="End result of the task",
                    artifact_metadata=None,
                    context=context,
                    generation=generation,
                )

            # Agent called one tool. We must call the tool and update the state.

            for tool_execution_suggestion in tool_call_generation.tool_calls:
                called_tools[tool_execution_suggestion] = available_tools[
                    tool_execution_suggestion.tool_name
                ].call(**tool_execution_suggestion.args)

            state.update(
                last_response=tool_call_generation.text,
                tool_calls_amount=tool_call_generation.tool_calls_amount(),
                iteration=state.iteration + 1,
                token_usage=tool_call_generation.usage,
            )

        raise MaxToolCallsExceededError(
            f"Max tool calls exceeded after {self.config.maxIterations} iterations"
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

    def _build_agent_run_output(
        self,
        *,
        artifacts: Sequence[Artifact] | None = None,
        artifact_name: str,
        artifact_description: str,
        artifact_metadata: dict[str, Any] | None,
        context: Context,
        generation: Generation[T_Schema],
        task_status: TaskState = TaskState.COMPLETED,
        append: bool = False,
        last_chunk: bool = False,
    ) -> AgentRunOutput[T_Schema]:
        return AgentRunOutput[T_Schema](
            artifacts=artifacts
            or [
                Artifact(
                    name=artifact_name,
                    description=artifact_description,
                    parts=list(
                        part for part in generation.parts if isinstance(part, TextPart)
                    ),
                    metadata=artifact_metadata,
                    index=0,
                    append=append,
                    last_chunk=last_chunk,
                )
            ],
            task_status=task_status,
            usage=AgentUsageStatistics(token_usage=generation.usage),
            final_context=context,
            parsed=generation.parsed,
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
        developer_message = DeveloperMessage(parts=[TextPart(text=instructions)])

        if isinstance(input, Context):
            # If it's already a Context, return it as is.
            return input
        elif isinstance(input, UserMessage):
            # If it's a UserMessage, prepend the developer instructions.
            return Context(messages=[developer_message, input])
        elif isinstance(input, str):
            # Handle plain string input
            return Context(
                messages=[
                    developer_message,
                    UserMessage(parts=[TextPart(text=input)]),
                ]
            )
        elif isinstance(input, (TextPart, FilePart, Tool)):
            # Handle single message parts
            return Context(
                messages=[
                    developer_message,
                    UserMessage(
                        parts=cast(Sequence[TextPart | FilePart | Tool], [input])
                    ),
                ]
            )
        elif callable(input) and not isinstance(input, Tool):
            # Handle callable input (that's not a Tool)
            return Context(
                messages=[
                    developer_message,
                    UserMessage(parts=[TextPart(text=input())]),
                ]
            )
        elif isinstance(input, pd.DataFrame):
            # Convert DataFrame to Markdown
            return Context(
                messages=[
                    developer_message,
                    UserMessage(parts=[TextPart(text=input.to_markdown() or "")]),
                ]
            )
        elif isinstance(input, np.ndarray):
            # Convert NumPy array to string representation
            return Context(
                messages=[
                    developer_message,
                    UserMessage(parts=[TextPart(text=np.array2string(input))]),
                ]
            )
        elif isinstance(input, Image.Image):
            import io

            img_byte_arr = io.BytesIO()
            img_format = getattr(input, "format", "PNG") or "PNG"
            input.save(img_byte_arr, format=img_format)
            img_byte_arr.seek(0)

            mime_type_map = {
                "PNG": "image/png",
                "JPEG": "image/jpeg",
                "JPG": "image/jpeg",
                "GIF": "image/gif",
                "WEBP": "image/webp",
                "BMP": "image/bmp",
                "TIFF": "image/tiff",
            }
            mime_type = mime_type_map.get(img_format, f"image/{img_format.lower()}")

            return Context(
                messages=[
                    developer_message,
                    UserMessage(
                        parts=[
                            FilePart(data=img_byte_arr.getvalue(), mime_type=mime_type)
                        ]
                    ),
                ]
            )
        elif isinstance(input, bytes):
            # Try decoding bytes, otherwise provide a description
            try:
                text = input.decode("utf-8")
            except UnicodeDecodeError:
                text = f"Input is binary data of size {len(input)} bytes."
            return Context(
                messages=[
                    developer_message,
                    UserMessage(parts=[TextPart(text=text)]),
                ]
            )
        elif isinstance(input, (dict, list, tuple, set, frozenset)):
            # Convert dict, list, tuple, set, frozenset to JSON string
            try:
                # Use json.dumps for serialization
                text = json.dumps(
                    input, indent=2, default=str
                )  # Add default=str for non-serializable
            except TypeError:
                # Fallback to string representation if json fails
                text = f"Input is a collection: {str(input)}"
            return Context(
                messages=[
                    developer_message,
                    UserMessage(parts=[TextPart(text=f"```json\n{text}\n```")]),
                ]
            )
        elif isinstance(input, (datetime.datetime, datetime.date, datetime.time)):
            # Convert datetime objects to ISO format string
            return Context(
                messages=[
                    developer_message,
                    UserMessage(parts=[TextPart(text=input.isoformat())]),
                ]
            )
        elif isinstance(input, Path):
            # Read file content if it's a file path that exists
            if input.is_file():
                try:
                    file_content = input.read_text()
                    return Context(
                        messages=[
                            developer_message,
                            UserMessage(parts=[TextPart(text=file_content)]),
                        ]
                    )
                except Exception as e:
                    # Fallback to string representation if reading fails
                    return Context(
                        messages=[
                            developer_message,
                            UserMessage(
                                parts=[
                                    TextPart(
                                        text=f"Falha ao ler o arquivo {input}: {str(e)}"
                                    )
                                ]
                            ),
                        ]
                    )
            else:
                # If it's not a file or doesn't exist, use the string representation
                return Context(
                    messages=[
                        developer_message,
                        UserMessage(parts=[TextPart(text=str(input))]),
                    ]
                )
        elif isinstance(input, (BytesIO, StringIO)):
            # Read content from BytesIO/StringIO
            input.seek(0)  # Ensure reading from the start
            content = input.read()
            if isinstance(content, bytes):
                try:
                    text = content.decode("utf-8")
                except UnicodeDecodeError:
                    text = f"Input is binary data stream of size {len(content)} bytes."
            else:  # str
                text = content
            return Context(
                messages=[
                    developer_message,
                    UserMessage(parts=[TextPart(text=text)]),
                ]
            )
        elif isinstance(input, PydanticBaseModel):
            # Convert Pydantic model to JSON string
            text = input.model_dump_json(indent=2)
            return Context(
                messages=[
                    developer_message,
                    UserMessage(parts=[TextPart(text=f"```json\n{text}\n```")]),
                ]
            )
        # Sequence handling: Check for Message sequences or Part sequences
        # Explicitly check for Sequence for MyPy's benefit
        elif isinstance(input, Sequence) and not isinstance(input, (str, bytes)):  # pyright: ignore[reportUnnecessaryIsInstance]
            # Check if it's a sequence of Messages or Parts (AFTER specific types)
            if input and isinstance(
                input[0], (AssistantMessage, DeveloperMessage, UserMessage)
            ):
                # Sequence of Messages
                # Ensure it's a list of Messages for type consistency
                return Context(messages=list(cast(Sequence[Message], input)))
            elif input and isinstance(input[0], (TextPart, FilePart, Tool)):
                # Sequence of Parts
                # Ensure it's a list of the correct Part types
                valid_parts = cast(Sequence[TextPart | FilePart | Tool], input)
                return Context(
                    messages=[
                        developer_message,
                        UserMessage(parts=list(valid_parts)),
                    ]
                )

        # Fallback for any unhandled type
        # Convert to string representation as a last resort
        return Context(
            messages=[
                developer_message,
                UserMessage(parts=[TextPart(text=str(input))]),
            ]
        )
