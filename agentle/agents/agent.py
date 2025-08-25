"""
The main module of the Agentle framework for creating and managing AI agents.

This module contains the definition of the Agent class, which is the central component of the Agentle framework.
It allows you to create intelligent agents capable of processing different types of input,
using external tools, and generating structured responses. The Agent facilitates integration
with different AI model providers and supports a wide variety of input formats.

Basic example:
```python
from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider
from agentle.agents.agent import Agent

weather_agent = Agent(
    generation_provider=GoogleGenerationProvider(),
    model="gemini-2.5-flash",
    instructions="You are a weather agent that can answer questions about the weather.",
    tools=[get_weather],
)

output = weather_agent.run("Hello. What is the weather in Tokyo?")
```
"""

# pyright: reportGeneralTypeIssues=false
# type: ignore[reportGeneralTypeIssues]

from __future__ import annotations

import base64
import datetime
import hashlib
import importlib.util
import json
import logging
import queue
import ssl
import threading
import time
import uuid
from collections.abc import (
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Generator,
    Iterator,
    Mapping,
    MutableMapping,
    MutableSequence,
    Sequence,
)
from contextlib import asynccontextmanager, contextmanager
from io import BytesIO, StringIO
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING, Any, Literal, cast, overload, override

import dill
from aiocache import cached
from rsb.containers.maybe import Maybe
from rsb.coroutines.run_sync import run_sync
from rsb.models.base_model import BaseModel
from rsb.models.config_dict import ConfigDict
from rsb.models.field import Field
from rsb.models.mimetype import MimeType

from agentle.agents.a2a.models.agent_skill import AgentSkill
from agentle.agents.a2a.models.authentication import Authentication
from agentle.agents.a2a.models.capabilities import Capabilities
from agentle.agents.a2a.models.run_state import RunState
from agentle.agents.agent_config import AgentConfig
from agentle.agents.agent_config_dict import AgentConfigDict
from agentle.agents.agent_input import AgentInput
from agentle.agents.agent_run_output import AgentRunOutput
from agentle.agents.apis.api import API
from agentle.agents.apis.endpoint import Endpoint
from agentle.agents.apis.endpoints_to_tools import endpoints_to_tools
from agentle.agents.context import Context
from agentle.agents.conversations.conversation_store import ConversationStore
from agentle.agents.errors.max_tool_calls_exceeded_error import (
    MaxToolCallsExceededError,
)
from agentle.agents.errors.tool_suspension_error import ToolSuspensionError
from agentle.agents.knowledge.static_knowledge import NO_CACHE, StaticKnowledge
from agentle.agents.message_history_fixer import MessageHistoryFixer
from agentle.agents.performance_metrics import PerformanceMetrics
from agentle.agents.step import Step
from agentle.agents.step_metric import StepMetric
from agentle.agents.suspension_manager import (
    SuspensionManager,
    get_default_suspension_manager,
)
from agentle.agents.ui.streamlit import AgentToStreamlit
from agentle.generations.collections.message_sequence import MessageSequence
from agentle.generations.models.generation.generation import Generation
from agentle.generations.models.generation.trace_params import TraceParams
from agentle.generations.models.message_parts.file import FilePart
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.message_parts.tool_execution_suggestion import (
    ToolExecutionSuggestion,
)
from agentle.generations.models.messages.assistant_message import AssistantMessage
from agentle.generations.models.messages.developer_message import DeveloperMessage
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.providers.base.generation_provider_mixin import (
    GenerationProviderMixin,
)
from agentle.generations.providers.base.generation_provider_type import (
    GenerationProviderType,
)
from agentle.generations.providers.google.google_generation_provider import (
    GoogleGenerationProvider,
)
from agentle.generations.providers.types.model_kind import ModelKind
from agentle.generations.tools.tool import Tool
from agentle.generations.tools.tool_execution_result import ToolExecutionResult
from agentle.mcp.servers.mcp_server_protocol import MCPServerProtocol
from agentle.parsing.cache.document_cache_store_type import DocumentCacheStoreType
from agentle.parsing.cache.in_memory_document_cache_store import (
    InMemoryDocumentCacheStore,
)

from agentle.parsing.factories.file_parser_default_factory import (
    file_parser_default_factory,
)
from agentle.parsing.parsed_file import ParsedFile
from agentle.parsing.parsers.document_parser_type import DocumentParserType
from agentle.prompts.models.prompt import Prompt
from agentle.stt.providers.base.speech_to_text_provider import SpeechToTextProvider
from agentle.vector_stores.vector_store import VectorStore

if TYPE_CHECKING:
    from blacksheep import Application
    from blacksheep.server.controllers import Controller
    from mcp.types import Tool as MCPTool

    from agentle.agents.agent_team import AgentTeam


type WithoutStructuredOutput = None
type _ToolName = str

logger = logging.getLogger(__name__)


# Check for optional dependencies
def is_module_available(module_name: str) -> bool:
    """Check if a module is available without importing it."""
    return importlib.util.find_spec(module_name) is not None


@dill.register(ssl.SSLContext)
def _save_sslcontext(pickler: Any, obj: ssl.SSLContext):  # type: ignore
    return ssl.create_default_context, ()


# Pre-check for common optional dependencies
HAS_PANDAS = is_module_available("pandas")
HAS_NUMPY = is_module_available("numpy")
HAS_PIL = is_module_available("PIL")
HAS_PYDANTIC = is_module_available("pydantic")


class Agent[T_Schema = WithoutStructuredOutput](BaseModel):
    """
    The main class of the Agentle framework that represents an intelligent agent.

    An Agent is an entity that can process various types of input,
    perform tasks using tools, and generate responses that can be structured.
    It encapsulates all the logic needed to interact with AI models,
    manage context, call external tools, and format responses.

    The Agent class is generic and supports structured response types through
    the T_Schema type parameter, which can be a Pydantic class to define
    the expected output structure.

    Attributes:
        name: Human-readable name of the agent.
        description: Description of the agent, used for communication with users and other agents.
        url: URL where the agent is hosted.
        generation_provider: Generation provider used by the agent.
        version: Version of the agent.
        endpoint: Endpoint of the agent.
        documentationUrl: URL to agent documentation.
        capabilities: Optional capabilities supported by the agent.
        authentication: Authentication requirements for the agent.
        defaultInputModes: Input interaction modes supported by the agent.
        defaultOutputModes: Output interaction modes supported by the agent.
        skills: Skills that the agent can perform.
        model: Model to be used by the agent's service provider.
        instructions: Instructions for the agent.
        response_schema: Schema of the response to be returned by the agent.
        mcp_servers: MCP servers to be used by the agent.
        tools: Tools to be used by the agent.
        config: Configuration for the agent.

    Example:
        ```python
        from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider
        from agentle.agents.agent import Agent

        # Define a simple tool
        def get_weather(location: str) -> str:
            return f"The weather in {location} is sunny."

        # Create a weather agent
        weather_agent = Agent(
            generation_provider=GoogleGenerationProvider(),
            model="gemini-2.5-flash",
            instructions="You are a weather agent that can answer questions about the weather.",
            tools=[get_weather],
        )

        # Run the agent
        output = weather_agent.run("What is the weather in London?")
        ```
    """

    uid: str = Field(default_factory=lambda: str(uuid.uuid4))
    """
    A unique identifier for the agent.
    """

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

    static_knowledge: Sequence[StaticKnowledge | str] = Field(default_factory=list)
    """
    Static knowledge to be used by the agent. This will be used to enrich the agent's
    knowledge base. This will be FULLY (**entire document**) indexed to the conversation.
    This can be any url or a local file path.
    
    You can provide a cache duration (in seconds) to cache the parsed content for subsequent calls.
    Example:
    ```python
    agent = Agent(
        static_knowledge=[
            StaticKnowledge(content="https://example.com/data.pdf", cache=3600),  # Cache for 1 hour
            StaticKnowledge(content="local_file.txt", cache="infinite"),  # Cache indefinitely
            "raw text knowledge"  # No caching (default)
        ]
    )
    ```
    """

    # Dear dev
    # Really sorry to use "Any" here. But if we use DocumentParser, we get an import cycle.
    # No worries, in the model_validator, we check if it's a DocumentParser.
    document_parser: DocumentParserType | None = Field(default=None)
    """
    A document parser to be used by the agent. This will be used to parse the static
    knowledge documents, if provided.
    """

    document_cache_store: DocumentCacheStoreType | None = Field(
        default=None,
        discriminator="type",
    )
    """
    A cache store to be used by the agent for caching parsed documents.
    If None, a default InMemoryDocumentCacheStore will be used.
    
    Example:
    ```python
    from agentle.parsing.cache import InMemoryDocumentCacheStore, RedisCacheStore
    
    # Use in-memory cache (default)
    agent = Agent(document_cache_store=InMemoryDocumentCacheStore())
    
    # Use Redis cache for distributed environments
    agent = Agent(document_cache_store=RedisCacheStore(redis_url="redis://localhost:6379/0"))
    ```
    """

    generation_provider: GenerationProviderType = Field(
        default_factory=GoogleGenerationProvider,
        discriminator="type",
    )
    """
    The service provider of the agent
    """

    file_visual_description_provider: GenerationProviderType | None = Field(
        default=None
    )
    """
    The service provider of the agent for visual description.
    """

    file_audio_description_provider: GenerationProviderType | None = Field(default=None)
    """
    The service provider of the agent for audio description.
    """

    version: str = Field(
        default="0.0.1",
        description="The version of the agent - format is up to the provider. (e.g. '1.0.0')",
        examples=["1.0.0", "1.0.1", "1.1.0"],
        pattern=r"^\d+\.\d+\.\d+$",
    )
    """
    The version of the agent - format is up to the provider. (e.g. "1.0.0")
    """

    endpoint: str | None = Field(
        default=None,
        description="The endpoint of the agent",
        examples=["/api/v1/agents/weather-agent"],
    )
    """
    The endpoint of the agent
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
    model: str | ModelKind | Callable[..., str] | None = Field(default=None)
    """
    The model to use for the agent's service provider.
    """

    instructions: str | Prompt | Callable[[], str] | MutableSequence[str] = Field(
        default="You are a helpful assistant."
    )
    """
    The instructions to use for the agent.
    """

    response_schema: type[T_Schema] | None = None
    """
    The schema of the response to be returned by the agent.
    """

    mcp_servers: MutableSequence[MCPServerProtocol] = Field(default_factory=list)
    """
    The MCP servers to use for the agent.
    """

    tools: MutableSequence[
        Tool | Callable[..., Any] | Callable[..., Awaitable[Any]]
    ] = Field(default_factory=list)
    """
    The tools to use for the agent.
    """

    config: AgentConfig | AgentConfigDict = Field(default_factory=AgentConfig)
    """
    The configuration for the agent.
    """

    debug: bool = Field(default=False)
    """
    Whether to debug each agent step using the logger.
    """

    suspension_manager: SuspensionManager | None = Field(default=None)
    """
    The suspension manager to use for Human-in-the-Loop workflows.
    If None, uses the default global suspension manager.
    """

    speech_to_text_provider: SpeechToTextProvider | None = Field(default=None)
    """
    The transcription provider to use for speech-to-text.
    """

    conversation_store: ConversationStore | None = Field(default=None)

    vector_stores: MutableSequence[VectorStore] | None = Field(default=None)

    endpoints: MutableSequence[Endpoint] = Field(
        default_factory=list,
        description="HTTP API endpoints that the agent can call. These will be automatically converted to tools.",
    )

    apis: MutableSequence[API] = Field(
        default_factory=list,
        description="Complete APIs with multiple endpoints that the agent can use. All endpoints in these APIs will be automatically converted to tools.",
    )

    # Internal fields
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def change_name(self, name: str | None = None) -> None:
        self.name = name or self.name

    def change_apis(self, apis: MutableSequence[API] | None = None) -> None:
        self.apis = apis or self.apis

    def change_endpoints(
        self, endpoints: MutableSequence[Endpoint] | None = None
    ) -> None:
        self.endpoints = endpoints or self.endpoints

    def change_instructions(
        self,
        instructions: str
        | Prompt
        | Callable[[], str]
        | MutableSequence[str]
        | None = None,
    ) -> None:
        self.instructions = instructions or self.instructions

    def append_instructions(
        self, instructions: str | Sequence[str] | None = None
    ) -> None:
        """
        Appends instructions to the existing agent instructions.

        Args:
            instructions: Instructions to append. Can be a string, sequence of strings, or None.
                        If None, no action is taken.
        """
        if instructions is None:
            return

        # Convert current instructions to a list of strings
        current_instructions: list[str] = []

        if isinstance(self.instructions, str):
            current_instructions = [self.instructions]
        elif isinstance(self.instructions, Prompt):
            current_instructions = [self.instructions.text]
        elif callable(self.instructions):
            current_instructions = [self.instructions()]
        else:
            # Must be MutableSequence[str] - convert all items to strings
            current_instructions = [str(item) for item in self.instructions]

        # Convert new instructions to list of strings
        if isinstance(instructions, str):
            new_instructions = [instructions]
        else:
            # Must be Sequence[str] - convert all items to strings
            new_instructions = [str(item) for item in instructions]

        # Combine and update
        current_instructions.extend(new_instructions)
        self.instructions = current_instructions

    @override
    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        _vs = self.vector_stores or []
        for vector_store in _vs:
            self.tools.append(vector_store.as_search_tool())

        # Collect all endpoints and APIs
        all_endpoints: MutableSequence[Endpoint | API] = []
        all_endpoints.extend(self.endpoints)
        all_endpoints.extend(self.apis)

        if all_endpoints:
            # Convert to tools
            api_tools = endpoints_to_tools(all_endpoints)

            # Add to existing tools
            self.tools.extend(api_tools)

            logger.debug(
                f"Converted {len(all_endpoints)} endpoints/APIs to {len(api_tools)} tools"
            )

    def add_endpoint(self, endpoint: Endpoint | Sequence[Endpoint]) -> None:
        if isinstance(endpoint, Sequence):
            self.endpoints.extend(endpoint)
            return

        self.endpoints.append(endpoint)

    def add_api(self, api: API | Sequence[API]) -> None:
        if isinstance(api, Sequence):
            self.apis.extend(api)
            return

        self.apis.append(api)

    def add_tool(self, tool: Callable[..., Any] | Callable[..., Awaitable[Any] | Tool]):
        self.tools += [tool]

    @property
    def agent_config(self) -> AgentConfig:
        if isinstance(self.config, dict):
            return AgentConfig.model_validate(self.config)

        return self.config

    @property
    def resolved_model(self) -> str | ModelKind | None:
        return (
            self.model()
            if callable(self.model)
            else self.model
            if isinstance(self.model, str)
            else None
        )

    def serialize(self) -> str:
        """
        Serializes the agent instance to a string.

        Returns:
            str: The serialized agent instance.
        """
        encoded: Mapping[str, Any] = {
            "version": "0.0.1",
            "uid": self.uid,
            "name": self.name,
            "instructions": self.instructions,
            "endpoints": self.endpoints,
            "apis": self.apis,
            # "tools": self.tools,
        }

        # pickle_bytes = dill.dumps(self)
        pickle_bytes = dill.dumps(encoded)

        # Encode to base64 and then decode to string
        base64_bytes = base64.b64encode(pickle_bytes)
        return base64_bytes.decode("utf-8")

    @classmethod
    def deserialize(cls, encoded: str) -> Agent[Any]:
        base64_bytes = encoded.encode("utf-8")
        pickle_bytes = base64.b64decode(base64_bytes)

        # Deserialize with dill
        obj = dill.loads(pickle_bytes)
        if isinstance(obj, dict):
            version = obj.get("version")
            if version is None:
                raise ValueError("Version not found")

            obj = cast(dict[str, Any], obj)

            match version:
                case "0.0.1":
                    return Agent(
                        name=obj.get("name") or "Agent",
                        uid=str(obj.get("uid")) or str(uuid.uuid4()),
                        instructions=obj.get("instructions")
                        or "You are a helpful assistant.",
                        endpoints=obj.get("endpoints") or [],
                        apis=obj.get("apis") or [],
                        tools=obj.get("tools") or [],
                    )
                case _:
                    raise NotImplementedError("Not implemented yet.")

        raise ValueError("not supported yet.")

    @cached(ttl=None)
    async def _all_tools(self) -> Sequence[Tool[Any]]:
        # Reconstruct the tool execution environment
        mcp_tools: Sequence[tuple[MCPServerProtocol, MCPTool]] = []
        if self.mcp_servers:
            for server in self.mcp_servers:
                tools = await server.list_tools_async()
                mcp_tools.extend((server, tool) for tool in tools)

        all_tools: Sequence[Tool[Any]] = [
            Tool.from_mcp_tool(mcp_tool=tool, server=server)
            for server, tool in mcp_tools
        ] + [
            Tool.from_callable(tool) if callable(tool) else tool for tool in self.tools
        ]

        return all_tools

    @classmethod
    def from_agent_card(cls, agent_card: Mapping[str, Any]) -> Agent[Any]:
        """
        Creates an Agent instance from an A2A agent card.

        This method parses an agent card dictionary and creates an Agent instance
        with the appropriate attributes. It maps the provider organization to a
        generation provider class if available.

        Args:
            agent_card: A dictionary representing an A2A agent card

        Returns:
            Agent[Any]: A new Agent instance based on the agent card

        Raises:
            KeyError: If a required field is missing from the agent card
            ValueError: If the provider organization is specified but not supported

        Example:
            ```python
            # Load an agent card from a file
            with open("agent_card.json", "r") as f:
                agent_card = json.load(f)

            # Create an agent from the card
            agent = Agent.from_agent_card(agent_card)
            ```
        """
        # Map provider organization to generation provider
        provider = agent_card.get("provider")
        generation_provider: Any = None

        if provider is not None:
            org_name = provider.get("organization", "")

            # Handle each provider type with proper error handling
            match org_name.lower():
                case "google":
                    # Default to Google provider if available
                    try:
                        from agentle.generations.providers.google import (
                            google_generation_provider,
                        )

                        generation_provider = (
                            google_generation_provider.GoogleGenerationProvider
                        )
                    except ImportError:
                        # Fail silently and use fallback later
                        pass
                case _:
                    raise ValueError(
                        f"Unsupported (yet) provider organization: {org_name}"
                    )

        # Convert skills
        skills: MutableSequence[AgentSkill] = []
        for skill_data in agent_card.get("skills", []):
            skill = AgentSkill(
                id=skill_data.get("id", str(uuid.uuid4())),
                name=skill_data["name"],
                description=skill_data["description"],
                tags=skill_data.get("tags", []),
                examples=skill_data.get("examples"),
                inputModes=skill_data.get("inputModes"),
                outputModes=skill_data.get("outputModes"),
            )
            skills.append(skill)

        # Create capabilities
        capabilities_data = agent_card.get("capabilities", {})
        capabilities = Capabilities(
            streaming=capabilities_data.get("streaming"),
            pushNotifications=capabilities_data.get("pushNotifications"),
            stateTransitionHistory=capabilities_data.get("stateTransitionHistory"),
        )

        # Create authentication
        auth_data = agent_card.get("authentication", {})
        authentication = Authentication(
            schemes=auth_data.get("schemes", ["basic"]),
            credentials=auth_data.get("credentials"),
        )

        # Default generation provider if none specified
        if generation_provider is None:
            try:
                from agentle.generations.providers.google import (
                    google_generation_provider,
                )

                generation_provider = (
                    google_generation_provider.GoogleGenerationProvider
                )
            except ImportError:
                # Create a minimal provider for type checking
                generation_provider = type(
                    "DummyProvider", (GenerationProviderMixin,), {}
                )

        # Convert input/output modes to MimeType if they're strings
        input_modes = agent_card.get("defaultInputModes", ["text/plain"])
        output_modes = agent_card.get("defaultOutputModes", ["text/plain"])

        # Create agent instance
        return cls(
            name=agent_card["name"],
            description=agent_card["description"],
            url=agent_card["url"],
            generation_provider=generation_provider,
            version=agent_card["version"],
            documentationUrl=agent_card.get("documentationUrl"),
            capabilities=capabilities,
            authentication=authentication,
            defaultInputModes=input_modes,
            defaultOutputModes=output_modes,
            skills=skills,
            # Default instructions based on description
            instructions=agent_card.get("description", "You are a helpful assistant."),
        )

    def to_agent_card(self) -> dict[str, Any]:
        """
        Generates an A2A agent card from this Agent instance.

        This method creates a dictionary representation of the agent in the A2A agent card
        format, including all relevant attributes such as name, description, capabilities,
        authentication, and skills.

        Returns:
            dict[str, Any]: A dictionary representing the A2A agent card

        Example:
            ```python
            # Create an agent
            agent = Agent(
                name="Weather Agent",
                description="An agent that provides weather information",
                generation_provider=GoogleGenerationProvider(),
                skills=[
                    AgentSkill(
                        name="Get Weather",
                        description="Gets the current weather for a location",
                        tags=["weather", "forecast"]
                    )
                ]
            )

            # Generate an agent card
            agent_card = agent.to_agent_card()

            # Save the agent card to a file
            with open("agent_card.json", "w") as f:
                json.dump(agent_card, f, indent=2)
            ```
        """
        # Determine provider information
        provider_dict: dict[str, str] | None = None
        provider_class = self.generation_provider.__class__

        # Map provider class to organization name
        provider_name: str | None = None
        provider_url = "https://example.com"  # just for now.

        if hasattr(provider_class, "__module__"):
            module_name = provider_class.__module__.lower()
            if "google" in module_name:
                provider_name = "Google"
                provider_url = "https://ai.google.dev/"  # just for now.
            elif "anthropic" in module_name:
                provider_name = "Anthropic"
                provider_url = "https://anthropic.com/"  # just for now.
            elif "openai" in module_name:
                provider_name = "OpenAI"
                provider_url = "https://openai.com/"  # just for now.

        if provider_name is not None:
            provider_dict = {"organization": provider_name, "url": provider_url}

        # Convert skills
        skills_data: MutableSequence[dict[str, Any]] = []
        for skill in self.skills:
            skill_data: dict[str, Any] = {
                "id": skill.id,
                "name": skill.name,
                "description": skill.description,
                "tags": list(skill.tags),
            }

            if skill.examples is not None:
                skill_data["examples"] = list(skill.examples)

            if skill.inputModes is not None:
                skill_data["inputModes"] = [str(mode) for mode in skill.inputModes]

            if skill.outputModes is not None:
                skill_data["outputModes"] = [str(mode) for mode in skill.outputModes]

            skills_data.append(skill_data)

        # Build capabilities dictionary
        capabilities: dict[str, bool] = {}
        if self.capabilities.streaming is not None:
            capabilities["streaming"] = self.capabilities.streaming
        if self.capabilities.pushNotifications is not None:
            capabilities["pushNotifications"] = self.capabilities.pushNotifications
        if self.capabilities.stateTransitionHistory is not None:
            capabilities["stateTransitionHistory"] = (
                self.capabilities.stateTransitionHistory
            )

        # Build authentication dictionary
        auth_dict: dict[str, Any] = {"schemes": list(self.authentication.schemes)}
        if self.authentication.credentials is not None:
            auth_dict["credentials"] = self.authentication.credentials

        # Convert MimeType to string for input/output modes
        input_modes = [str(mode) for mode in self.defaultInputModes]
        output_modes = [str(mode) for mode in self.defaultOutputModes]

        # Build agent card
        agent_card: dict[str, Any] = {
            "name": self.name,
            "description": self.description,
            "url": self.url,
            "version": self.version,
            "capabilities": capabilities,
            "authentication": auth_dict,
            "defaultInputModes": input_modes,
            "defaultOutputModes": output_modes,
            "skills": skills_data,
        }

        # Add optional fields if they exist
        if provider_dict is not None:
            agent_card["provider"] = provider_dict

        if self.documentationUrl is not None:
            agent_card["documentationUrl"] = self.documentationUrl

        return agent_card

    def has_tools(self) -> bool:
        """
        Checks if this agent has configured tools.

        Returns:
            bool: True if the agent has tools, False otherwise.
        """
        return len(self.tools) > 0

    @contextmanager
    def start_mcp_servers(self) -> Generator[None, None, None]:
        """
        Context manager to connect and clean up MCP servers.

        This context manager ensures that all MCP servers are connected before the
        code block is executed and cleaned up after completion, even in case of exceptions.

        Yields:
            None: Does not return a value, just manages the context.

        Example:
            ```python
            async with agent.start_mcp_servers():
                # Operations that require connection to MCP servers
                result = await agent.run_async("Query to server")
            # Servers are automatically disconnected here
            ```
        """
        for server in self.mcp_servers:
            server.connect()
        try:
            yield
        finally:
            for server in self.mcp_servers:
                server.cleanup()

    @asynccontextmanager
    async def start_mcp_servers_async(self) -> AsyncGenerator[None, None]:
        """
        Asynchronous context manager to connect and clean up MCP servers.

        This method ensures that all MCP servers are connected before the
        code block is executed and cleaned up after completion, even in case of exceptions.

        Yields:
            None: Does not return a value, just manages the context.

        Example:
            ```python
            async with agent.start_mcp_servers():
                # Operations that require connection to MCP servers
                result = await agent.run_async("Query to server")
            # Servers are automatically disconnected here
            ```
        """
        for server in self.mcp_servers:
            await server.connect_async()
        try:
            yield
        finally:
            for server in self.mcp_servers:
                await server.cleanup_async()

    async def resume_async(
        self, resumption_token: str, approval_data: dict[str, Any] | None = None
    ) -> AgentRunOutput[T_Schema]:
        """
        Resume a suspended agent execution.

        Args:
            resumption_token: Token from a suspended execution
            approval_data: Optional approval data to pass to the resumed execution

        Returns:
            AgentRunOutput with the completed or newly suspended execution

        Raises:
            ValueError: If the resumption token is invalid or expired
        """
        suspension_manager = get_default_suspension_manager()

        # Resume the execution
        result = await suspension_manager.resume_execution(
            resumption_token, approval_data
        )

        if result is None:
            raise ValueError(f"Invalid or expired resumption token: {resumption_token}")

        context, _ = result

        # Continue execution from where it left off
        return await self._continue_execution_from_context(context)

    def resume(
        self, resumption_token: str, approval_data: dict[str, Any] | None = None
    ) -> AgentRunOutput[T_Schema]:
        """
        Resume a suspended agent execution synchronously.

        Args:
            resumption_token: Token from a suspended execution
            approval_data: Optional approval data to pass to the resumed execution

        Returns:
            AgentRunOutput with the completed or newly suspended execution
        """
        return run_sync(
            self.resume_async,
            resumption_token=resumption_token,
            approval_data=approval_data,
        )

    @overload
    def run(
        self,
        input: AgentInput | Any,
        *,
        timeout: float | None = None,
        trace_params: TraceParams | None = None,
        chat_id: str | None = None,
        stream: Literal[False] = False,
    ) -> AgentRunOutput[T_Schema]: ...

    @overload
    def run(
        self,
        input: AgentInput | Any,
        *,
        timeout: float | None = None,
        trace_params: TraceParams | None = None,
        chat_id: str | None = None,
        stream: Literal[True],
    ) -> Iterator[AgentRunOutput[T_Schema]]: ...

    def run(
        self,
        input: AgentInput | Any,
        *,
        timeout: float | None = None,
        trace_params: TraceParams | None = None,
        chat_id: str | None = None,
        stream: bool = False,
    ) -> AgentRunOutput[T_Schema] | Iterator[AgentRunOutput[T_Schema]]:
        """
        Runs the agent synchronously with the provided input.

        This method is a synchronous wrapper for run_async, allowing
        easy use in synchronous contexts. When streaming is enabled,
        it returns an iterator that yields chunks synchronously.

        Args:
            input: The input for the agent, which can be of various types.
            timeout: Optional time limit in seconds for execution.
            trace_params: Optional trace parameters for observability purposes.
            chat_id: Optional chat ID for conversation persistence.
            stream: Whether to stream responses. If True, returns an iterator.

        Returns:
            AgentRunOutput[T_Schema] | Iterator[AgentRunOutput[T_Schema]]:
                Single result or iterator of streaming chunks.

        Example:
            ```python
            # Non-streaming
            result = agent.run("What is the weather in London?")

            # Streaming
            for chunk in agent.run("What is the weather in London?", stream=True):
                print(chunk.generation.text)

            # Input as UserMessage object
            from agentle.generations.models.messages.user_message import UserMessage
            from agentle.generations.models.message_parts.text import TextPart

            message = UserMessage(parts=[TextPart(text="What is the weather in London?")])
            result = agent.run(message)
            ```
        """
        if stream:
            # Create a sync iterator that bridges the async iterator
            def _sync_stream_iterator() -> Iterator[AgentRunOutput[T_Schema]]:
                # Queue to store chunks as they arrive

                chunk_queue: queue.Queue[AgentRunOutput[T_Schema] | None] = (
                    queue.Queue()
                )
                exception_holder: list[Exception] = []

                # Background thread to consume async iterator
                async def _consume_async_iterator():
                    try:
                        async for chunk in await self.run_async(
                            input=input,
                            trace_params=trace_params,
                            chat_id=chat_id,
                            stream=True,
                        ):
                            chunk_queue.put(chunk)
                        chunk_queue.put(None)  # Signal end
                    except Exception as e:
                        exception_holder.append(e)
                        chunk_queue.put(None)  # Signal end

                # Start the async consumption in background
                def run_async_consumer():
                    run_sync(_consume_async_iterator, timeout=timeout)

                consumer_thread = threading.Thread(target=run_async_consumer)
                consumer_thread.start()

                # Yield chunks as they become available
                try:
                    while True:
                        if exception_holder:
                            raise exception_holder[0]

                        try:
                            chunk = chunk_queue.get(timeout=timeout or 30)
                            if chunk is None:  # End signal
                                break
                            yield chunk
                        except queue.Empty:
                            if exception_holder:
                                raise exception_holder[0]
                            raise TimeoutError("Timeout waiting for next chunk")
                finally:
                    consumer_thread.join(timeout=1)

            return _sync_stream_iterator()

        return run_sync(
            self.run_async,
            timeout=timeout,
            input=input,
            trace_params=trace_params,
            chat_id=chat_id,
            stream=stream,
        )

    @overload
    async def run_async(
        self,
        input: AgentInput | Any,
        *,
        trace_params: TraceParams | None = None,
        chat_id: str | None = None,
        stream: Literal[False] = False,
    ) -> AgentRunOutput[T_Schema]: ...

    @overload
    async def run_async(
        self,
        input: AgentInput | Any,
        *,
        trace_params: TraceParams | None = None,
        chat_id: str | None = None,
        stream: Literal[True],
    ) -> AsyncIterator[AgentRunOutput[T_Schema]]: ...

    async def run_async(
        self,
        input: AgentInput | Any,
        *,
        trace_params: TraceParams | None = None,
        chat_id: str | None = None,
        stream: bool = False,
    ) -> AgentRunOutput[T_Schema] | AsyncIterator[AgentRunOutput[T_Schema]]:
        """
        Runs the agent asynchronously with the provided input and collects comprehensive performance metrics.

        Now supports streaming responses for real-time output generation.

        This main method processes user input, interacts with the
        generation provider, and optionally calls tools until reaching a final response.
        It now includes detailed performance monitoring to help identify optimization opportunities.

        The method supports both simple agents (without tools) and agents with
        tools that can perform iterative calls to solve complex tasks.

        Args:
            input: The input for the agent, which can be of various types.
            trace_params: Optional trace parameters for observability purposes.
            chat_id: Optional chat ID for conversation persistence.
            stream: Whether to stream the response. Requires provider to support streaming.

        Returns:
            AgentRunOutput[T_Schema] | AsyncIterator[AgentRunOutput[T_Schema]]:
                The result of the agent execution with performance metrics,
                possibly with a structured response according to the defined schema.
                If streaming is enabled, returns an async iterator of partial results.

        Raises:
            MaxToolCallsExceededError: If the maximum number of tool calls is exceeded.
            ValueError: If streaming is requested but the provider doesn't support it.
        """
        import time

        from agentle.generations.providers.base.supports_streaming import (
            SupportsStreaming,
        )

        # Check if streaming is supported when requested
        if stream and not isinstance(self.generation_provider, SupportsStreaming):
            raise ValueError(
                f"Streaming is not supported by {type(self.generation_provider).__name__}. "
                + "The provider must implement the SupportsStreaming interface."
            )

        # Start overall timing
        execution_start_time = time.perf_counter()

        # Initialize metrics tracking
        step_metrics: list[StepMetric] = []
        generation_time_total = 0.0
        tool_execution_time_total = 0.0
        iteration_count = 0
        tool_calls_count = 0
        total_tokens_processed = 0
        cache_hits = 0
        cache_misses = 0

        _logger = Maybe(logger if self.debug else None)

        if chat_id is not None and self.conversation_store is None:
            raise ValueError(
                "Chat ID was provided but no conversation store was "
                + "provided in the Agent's constructor."
            )

        # Phase 1: Input Processing
        input_processing_start = time.perf_counter()

        _logger.bind_optional(
            lambda log: log.info(
                "Starting agent run with input type: %s (streaming=%s)",
                str(type(input).__name__),
                stream,
            )
        )
        generation_provider: GenerationProviderType = self.generation_provider

        static_knowledge_prompt: str | None = None

        input_processing_time = (time.perf_counter() - input_processing_start) * 1000

        # Phase 2: Static Knowledge Processing
        static_knowledge_start = time.perf_counter()

        # Process static knowledge if any exists
        if self.static_knowledge:
            _logger.bind_optional(lambda log: log.debug("Processing static knowledge"))
            knowledge_contents: MutableSequence[str] = []

            # Get or create cache store
            document_cache_store = (
                self.document_cache_store or InMemoryDocumentCacheStore()
            )

            for knowledge_item in self.static_knowledge:
                # Convert string to StaticKnowledge with NO_CACHE
                if isinstance(knowledge_item, str):
                    knowledge_item = StaticKnowledge(
                        content=knowledge_item, cache=NO_CACHE, parse_timeout=30
                    )

                # Process the knowledge item based on its content type
                content_to_parse = knowledge_item.content
                parsed_content = None
                parser = self.document_parser or file_parser_default_factory(
                    visual_description_provider=generation_provider
                    if self.file_visual_description_provider is None
                    else self.file_visual_description_provider,
                    audio_description_provider=generation_provider
                    if self.file_audio_description_provider is None
                    else self.file_audio_description_provider,
                    parse_timeout=knowledge_item.parse_timeout,
                )

                # Check if caching is enabled
                if knowledge_item.cache is not NO_CACHE:
                    _logger.bind_optional(
                        lambda log: log.debug("Using cache store for knowledge item")
                    )

                    cache_key = hashlib.sha256(
                        f"{content_to_parse}:{parser.__class__.__name__}".encode()
                    ).hexdigest()

                    # Try to get from cache first
                    parsed_content = await document_cache_store.get_async(cache_key)

                    if parsed_content is None:
                        # Not in cache, parse and store
                        cache_misses += 1
                        _logger.bind_optional(
                            lambda log: log.debug("Cache miss, parsing and storing")
                        )
                        if knowledge_item.is_url():
                            parsed_content = await parser.parse_async(content_to_parse)
                        elif knowledge_item.is_file_path():
                            parsed_content = await parser.parse_async(content_to_parse)
                        else:  # Raw text - don't cache raw text
                            parsed_content = None

                        # Store in cache if we parsed something
                        if parsed_content is not None:
                            await document_cache_store.set_async(
                                cache_key, parsed_content, ttl=knowledge_item.cache
                            )
                    else:
                        cache_hits += 1
                        _logger.bind_optional(
                            lambda log: log.debug("Cache hit for knowledge item")
                        )

                # If no cached content (either cache not enabled or cache miss), parse directly
                if parsed_content is None:
                    if knowledge_item.is_url():
                        _logger.bind_optional(
                            lambda log: log.debug("Parsing URL: %s", content_to_parse)
                        )
                        parsed_content = await parser.parse_async(content_to_parse)
                        knowledge_contents.append(
                            f"## URL: {content_to_parse}\n\n{parsed_content.sections[0].text}"
                        )
                    elif knowledge_item.is_file_path():
                        _logger.bind_optional(
                            lambda log: log.debug("Parsing file: %s", content_to_parse)
                        )
                        parsed_content = await parser.parse_async(content_to_parse)
                        knowledge_contents.append(
                            f"## Document: {parsed_content.name}\n\n{parsed_content.sections[0].text}"
                        )
                    else:  # Raw text
                        _logger.bind_optional(
                            lambda log: log.debug("Using raw text knowledge")
                        )
                        knowledge_contents.append(
                            f"## Information:\n\n{content_to_parse}"
                        )
                else:
                    # Use the cached content
                    source_label = (
                        "URL"
                        if knowledge_item.is_url()
                        else "Document"
                        if knowledge_item.is_file_path()
                        else "Information"
                    )
                    knowledge_contents.append(
                        f"## {source_label}: {content_to_parse}\n\n{parsed_content.sections[0].text}"
                    )

            if knowledge_contents:
                static_knowledge_prompt = "\n\n# KNOWLEDGE BASE\n\n" + "\n\n".join(
                    knowledge_contents
                )

        static_knowledge_time = (time.perf_counter() - static_knowledge_start) * 1000

        instructions = self.instructions2str(self.instructions)
        if static_knowledge_prompt:
            instructions += "\n\n" + static_knowledge_prompt

        # Create context with current input
        context: Context = self.input2context(input, instructions=instructions)
        context.message_history = list(
            MessageHistoryFixer().fix_message_history(context.message_history)
        )

        # CRITICAL: Store the current user message BEFORE replacing message history
        current_user_message = context.last_message

        # Handle conversation store integration
        if chat_id:
            assert self.conversation_store is not None

            # Get existing conversation history
            conversation_history = (
                await self.conversation_store.get_conversation_history_async(chat_id)
            )

            # Replace message history with conversation history, keeping developer messages
            if conversation_history:
                context.replace_message_history(
                    conversation_history, keep_developer_messages=True
                )
                # Add the current user message to the context
                context.message_history.append(current_user_message)

        # Start execution tracking
        context.start_execution()

        _logger.bind_optional(
            lambda log: log.debug(
                "Converted input to context with %d messages",
                len(context.message_history),
            )
        )

        _logger.bind_optional(
            lambda log: log.debug(
                "Using generation provider: %s", type(generation_provider).__name__
            )
        )

        # Phase 3: MCP Tools Preparation
        mcp_tools_start = time.perf_counter()

        mcp_tools: MutableSequence[tuple[MCPServerProtocol, MCPTool]] = []
        if bool(self.mcp_servers):
            _logger.bind_optional(
                lambda log: log.debug("Getting tools from MCP servers")
            )
            for server in self.mcp_servers:
                tools = await server.list_tools_async()
                mcp_tools.extend((server, tool) for tool in tools)
            _logger.bind_optional(
                lambda log: log.debug("Got %d tools from MCP servers", len(mcp_tools))
            )

        mcp_tools_time = (time.perf_counter() - mcp_tools_start) * 1000

        agent_has_tools = self.has_tools() or len(mcp_tools) > 0
        _logger.bind_optional(
            lambda log: log.debug("Agent has tools: %s", agent_has_tools)
        )

        # Handle non-tool agents (simpler case)
        if not agent_has_tools:
            _logger.bind_optional(
                lambda log: log.debug("No tools available, generating direct response")
            )

            if stream and isinstance(generation_provider, SupportsStreaming):
                # Streaming path for non-tool agents
                _logger.bind_optional(
                    lambda log: log.debug(
                        "Using streaming generation for direct response"
                    )
                )

                async def _stream_direct_response() -> AsyncIterator[
                    AgentRunOutput[T_Schema]
                ]:
                    nonlocal generation_time_total, total_tokens_processed

                    step_start_time = time.perf_counter()
                    step = Step(
                        step_type="generation",
                        iteration=1,
                        tool_execution_suggestions=[],
                        generation_text="Generating streaming response...",
                        token_usage=None,
                    )

                    generation_start = time.perf_counter()

                    # Stream the generation
                    chunk_count = 0
                    final_generation: Generation[T_Schema] | None = None

                    async for generation_chunk in cast(
                        SupportsStreaming, generation_provider
                    ).stream_async(
                        model=self.resolved_model,
                        messages=context.message_history,
                        response_schema=self.response_schema,
                        generation_config=self.agent_config.generation_config
                        if trace_params is None
                        else self.agent_config.generation_config.clone(
                            new_trace_params=trace_params
                        ),
                    ):
                        generation_chunk = cast(Generation[T_Schema], generation_chunk)
                        chunk_count += 1
                        generation_time_single = (
                            time.perf_counter() - generation_start
                        ) * 1000

                        # Update metrics
                        if generation_chunk.usage:
                            total_tokens_processed = generation_chunk.usage.total_tokens

                        # Create partial performance metrics
                        partial_metrics = PerformanceMetrics(
                            total_execution_time_ms=(
                                time.perf_counter() - execution_start_time
                            )
                            * 1000,
                            input_processing_time_ms=input_processing_time,
                            static_knowledge_processing_time_ms=static_knowledge_time,
                            mcp_tools_preparation_time_ms=mcp_tools_time,
                            generation_time_ms=generation_time_single,
                            tool_execution_time_ms=0.0,
                            final_response_processing_time_ms=0.0,
                            iteration_count=1,
                            tool_calls_count=0,
                            total_tokens_processed=total_tokens_processed,
                            cache_hit_rate=(
                                cache_hits / (cache_hits + cache_misses) * 100
                            )
                            if (cache_hits + cache_misses) > 0
                            else 0.0,
                            step_metrics=[],
                            average_generation_time_ms=generation_time_single,
                            average_tool_execution_time_ms=0.0,
                            longest_step_duration_ms=generation_time_single,
                            shortest_step_duration_ms=generation_time_single,
                        )

                        # Yield intermediate chunk
                        yield AgentRunOutput(
                            generation=generation_chunk,
                            context=context,
                            parsed=generation_chunk.parsed
                            if hasattr(generation_chunk, "parsed")
                            else cast(T_Schema, None),
                            is_streaming_chunk=True,
                            is_final_chunk=False,
                            performance_metrics=partial_metrics,
                        )

                        final_generation = generation_chunk

                    # After streaming completes
                    if final_generation:
                        context.message_history.append(
                            final_generation.message.to_assistant_message()
                        )
                        generation_time_total = (
                            time.perf_counter() - generation_start
                        ) * 1000

                        _logger.bind_optional(
                            lambda log: log.debug(
                                "Streamed response with %d chunks, %d tokens",
                                chunk_count,
                                final_generation.usage.total_tokens,
                            )
                        )

                        # Update step with final results
                        step.generation_text = final_generation.text
                        step.token_usage = final_generation.usage
                        step_duration = (time.perf_counter() - step_start_time) * 1000
                        step.mark_completed(duration_ms=step_duration)

                        # Track step metrics
                        step_metrics.append(
                            StepMetric(
                                step_id=step.step_id,
                                step_type="generation",
                                duration_ms=step_duration,
                                iteration=1,
                                tool_calls_count=0,
                                generation_tokens=final_generation.usage.total_tokens,
                                cache_hits=0,
                                cache_misses=0,
                            )
                        )

                        # Add step to context
                        context.add_step(step)
                        context.update_token_usage(final_generation.usage)
                        context.complete_execution()

                        total_tokens_processed = final_generation.usage.total_tokens
                        total_execution_time = (
                            time.perf_counter() - execution_start_time
                        ) * 1000

                        # Create final performance metrics
                        performance_metrics = PerformanceMetrics(
                            total_execution_time_ms=total_execution_time,
                            input_processing_time_ms=input_processing_time,
                            static_knowledge_processing_time_ms=static_knowledge_time,
                            mcp_tools_preparation_time_ms=mcp_tools_time,
                            generation_time_ms=generation_time_total,
                            tool_execution_time_ms=0.0,
                            final_response_processing_time_ms=0.0,
                            iteration_count=1,
                            tool_calls_count=0,
                            total_tokens_processed=total_tokens_processed,
                            cache_hit_rate=(
                                cache_hits / (cache_hits + cache_misses) * 100
                            )
                            if (cache_hits + cache_misses) > 0
                            else 0.0,
                            step_metrics=step_metrics,
                            average_generation_time_ms=generation_time_total,
                            average_tool_execution_time_ms=0.0,
                            longest_step_duration_ms=step_duration,
                            shortest_step_duration_ms=step_duration,
                        )

                        # Save to conversation store after successful execution
                        if chat_id:
                            assert self.conversation_store is not None
                            await self.conversation_store.add_message_async(
                                chat_id, current_user_message
                            )
                            await self.conversation_store.add_message_async(
                                chat_id, final_generation.message.to_assistant_message()
                            )

                        # Yield final chunk
                        yield AgentRunOutput(
                            generation=final_generation,
                            context=context,
                            parsed=final_generation.parsed,
                            is_streaming_chunk=False,
                            is_final_chunk=True,
                            performance_metrics=performance_metrics,
                        )

                # Create and return the async generator
                stream_generator = _stream_direct_response()
                return stream_generator

            # Non-streaming path (existing code)
            step_start_time = time.perf_counter()
            step = Step(
                step_type="generation",
                iteration=1,
                tool_execution_suggestions=[],
                generation_text="Generating direct response...",
                token_usage=None,
            )

            generation_start = time.perf_counter()
            generation: Generation[T_Schema] = await generation_provider.generate_async(
                model=self.resolved_model,
                messages=context.message_history,
                response_schema=self.response_schema,
                generation_config=self.agent_config.generation_config
                if trace_params is None
                else self.agent_config.generation_config.clone(
                    new_trace_params=trace_params
                ),
            )
            context.message_history.append(generation.message.to_assistant_message())
            generation_time_single = (time.perf_counter() - generation_start) * 1000
            generation_time_total += generation_time_single

            _logger.bind_optional(
                lambda log: log.debug(
                    "Generated response with %d tokens", generation.usage.total_tokens
                )
            )

            # Update the step with the actual generation results
            step.generation_text = generation.text
            step.token_usage = generation.usage
            step_duration = (time.perf_counter() - step_start_time) * 1000
            step.mark_completed(duration_ms=step_duration)

            # Track step metrics
            step_metrics.append(
                StepMetric(
                    step_id=step.step_id,
                    step_type="generation",
                    duration_ms=step_duration,
                    iteration=1,
                    tool_calls_count=0,
                    generation_tokens=generation.usage.total_tokens,
                    cache_hits=0,
                    cache_misses=0,
                )
            )

            # Add the step to context
            context.add_step(step)

            # Update context with final generation and complete execution
            context.update_token_usage(generation.usage)
            context.complete_execution()

            total_tokens_processed = generation.usage.total_tokens
            total_execution_time = (time.perf_counter() - execution_start_time) * 1000
            final_response_processing_time = 0.0

            # Create performance metrics
            performance_metrics = PerformanceMetrics(
                total_execution_time_ms=total_execution_time,
                input_processing_time_ms=input_processing_time,
                static_knowledge_processing_time_ms=static_knowledge_time,
                mcp_tools_preparation_time_ms=mcp_tools_time,
                generation_time_ms=generation_time_total,
                tool_execution_time_ms=0.0,
                final_response_processing_time_ms=final_response_processing_time,
                iteration_count=1,
                tool_calls_count=0,
                total_tokens_processed=total_tokens_processed,
                cache_hit_rate=(cache_hits / (cache_hits + cache_misses) * 100)
                if (cache_hits + cache_misses) > 0
                else 0.0,
                step_metrics=step_metrics,
                average_generation_time_ms=generation_time_total,
                average_tool_execution_time_ms=0.0,
                longest_step_duration_ms=step_duration,
                shortest_step_duration_ms=step_duration,
            )

            # Save to conversation store after successful execution
            if chat_id:
                assert self.conversation_store is not None
                await self.conversation_store.add_message_async(
                    chat_id, current_user_message
                )
                await self.conversation_store.add_message_async(
                    chat_id, generation.message.to_assistant_message()
                )

            return AgentRunOutput(
                generation=generation,
                context=context,
                parsed=generation.parsed,
                performance_metrics=performance_metrics,
            )

        # Agent has tools - handle streaming and non-streaming paths
        all_tools: Sequence[Tool] = cast(Sequence[Tool], await self._all_tools())

        available_tools: MutableMapping[str, Tool[Any]] = {
            tool.name: tool for tool in all_tools
        }

        state = RunState[T_Schema].init_state()

        # FIXED: Track tool suggestions and results for deduplication only
        all_tool_results: dict[str, tuple[ToolExecutionSuggestion, Any]] = {}
        tool_call_patterns: dict[str, int] = {}
        tool_budget: dict[str, int] = {}

        # For streaming with tools, we need a different approach
        if stream and isinstance(generation_provider, SupportsStreaming):
            _logger.bind_optional(
                lambda log: log.debug("Using streaming generation with tools")
            )

            async def _stream_with_tools() -> AsyncIterator[AgentRunOutput[T_Schema]]:
                nonlocal state, all_tool_results, tool_call_patterns, tool_budget
                nonlocal \
                    generation_time_total, \
                    tool_execution_time_total, \
                    iteration_count, \
                    tool_calls_count, \
                    total_tokens_processed

                while state.iteration < self.agent_config.maxIterations:
                    current_iteration = state.iteration + 1
                    iteration_count = current_iteration

                    _logger.bind_optional(
                        lambda log: log.info(
                            "Starting streaming iteration %d of %d",
                            current_iteration,
                            self.agent_config.maxIterations,
                        )
                    )

                    # FIXED: Use context message history directly
                    message_history = context.message_history

                    _logger.bind_optional(
                        lambda log: log.debug("Streaming tool call generation")
                    )

                    generation_start = time.perf_counter()

                    # Stream tool call generation (no response_schema for tool calls)
                    final_tool_generation: (
                        Generation[WithoutStructuredOutput] | None
                    ) = None
                    chunk_count = 0

                    async for tool_generation_chunk in generation_provider.stream_async(
                        model=self.resolved_model,
                        messages=message_history,
                        generation_config=self.agent_config.generation_config,
                        tools=all_tools,
                    ):
                        chunk_count += 1
                        final_tool_generation = tool_generation_chunk

                        # We don't yield intermediate chunks for tool calls
                        # since we need to execute the tools first

                    if final_tool_generation is None:
                        break

                    final_tool_generation = cast(
                        Generation[WithoutStructuredOutput], final_tool_generation
                    )

                    generation_time_single = (
                        time.perf_counter() - generation_start
                    ) * 1000
                    generation_time_total += generation_time_single

                    _logger.bind_optional(
                        lambda log: log.debug(
                            "Tool call streaming completed with %d chunks, %d tool calls",
                            chunk_count,
                            cast(
                                Generation[WithoutStructuredOutput],
                                final_tool_generation,
                            ).tool_calls_amount(),
                        )
                    )

                    # Update context with token usage from this generation
                    context.update_token_usage(final_tool_generation.usage)
                    total_tokens_processed += final_tool_generation.usage.total_tokens

                    agent_didnt_call_any_tool = (
                        final_tool_generation.tool_calls_amount() == 0
                    )
                    if agent_didnt_call_any_tool:
                        _logger.bind_optional(
                            lambda log: log.info(
                                "Agent didn't call any tool, generating final streaming response"
                            )
                        )

                        # Create a step for the final generation
                        final_step_start_time = time.perf_counter()
                        final_step = Step(
                            step_type="generation",
                            iteration=current_iteration,
                            tool_execution_suggestions=[],
                            generation_text=final_tool_generation.text
                            or "Generating final response...",
                            token_usage=final_tool_generation.usage,
                        )

                        # Only make another call if we need structured output or didn't get text
                        if (
                            self.response_schema is not None
                            or not final_tool_generation.text
                        ):
                            _logger.bind_optional(
                                lambda log: log.debug("Streaming structured response")
                            )

                            # Stream final structured response (no tools for structured output)
                            generation_start = time.perf_counter()
                            final_generation: Generation[T_Schema] | None = None
                            chunk_count = 0

                            async for (
                                generation_chunk
                            ) in generation_provider.stream_async(
                                model=self.resolved_model,
                                messages=list(context.message_history)
                                + [
                                    final_tool_generation.message.to_assistant_message()
                                ],
                                response_schema=self.response_schema,
                                generation_config=self.agent_config.generation_config,
                            ):
                                generation_chunk = cast(
                                    Generation[T_Schema], generation_chunk
                                )
                                chunk_count += 1
                                generation_time_single = (
                                    time.perf_counter() - generation_start
                                ) * 1000

                                # Update metrics
                                if generation_chunk.usage:
                                    total_tokens_processed += (
                                        generation_chunk.usage.completion_tokens
                                    )

                                # Create partial performance metrics
                                partial_metrics = PerformanceMetrics(
                                    total_execution_time_ms=(
                                        time.perf_counter() - execution_start_time
                                    )
                                    * 1000,
                                    input_processing_time_ms=input_processing_time,
                                    static_knowledge_processing_time_ms=static_knowledge_time,
                                    mcp_tools_preparation_time_ms=mcp_tools_time,
                                    generation_time_ms=generation_time_total
                                    + generation_time_single,
                                    tool_execution_time_ms=tool_execution_time_total,
                                    final_response_processing_time_ms=0.0,
                                    iteration_count=iteration_count,
                                    tool_calls_count=tool_calls_count,
                                    total_tokens_processed=total_tokens_processed,
                                    cache_hit_rate=(
                                        cache_hits / (cache_hits + cache_misses) * 100
                                    )
                                    if (cache_hits + cache_misses) > 0
                                    else 0.0,
                                    step_metrics=step_metrics,
                                    average_generation_time_ms=(
                                        generation_time_total + generation_time_single
                                    )
                                    / max(1, iteration_count),
                                    average_tool_execution_time_ms=tool_execution_time_total
                                    / max(1, tool_calls_count),
                                    longest_step_duration_ms=max(
                                        [s.duration_ms for s in step_metrics],
                                        default=0.0,
                                    ),
                                    shortest_step_duration_ms=min(
                                        [s.duration_ms for s in step_metrics],
                                        default=0.0,
                                    ),
                                )

                                # Yield intermediate chunk
                                yield AgentRunOutput(
                                    generation=generation_chunk,
                                    context=context,
                                    parsed=generation_chunk.parsed
                                    if hasattr(generation_chunk, "parsed")
                                    else cast(T_Schema, None),
                                    is_streaming_chunk=True,
                                    is_final_chunk=False,
                                    performance_metrics=partial_metrics,
                                )

                                final_generation = generation_chunk

                            if final_generation:
                                context.message_history.append(
                                    final_generation.message.to_assistant_message()
                                )
                                generation_time_single = (
                                    time.perf_counter() - generation_start
                                ) * 1000
                                generation_time_total += generation_time_single

                                _logger.bind_optional(
                                    lambda log: log.debug(
                                        "Final streaming generation complete with %d chunks",
                                        chunk_count,
                                    )
                                )

                                # Update the step with the final generation results
                                final_step.generation_text = final_generation.text
                                final_step.token_usage = final_generation.usage
                                final_step_duration = (
                                    time.perf_counter() - final_step_start_time
                                ) * 1000
                                final_step.mark_completed(
                                    duration_ms=final_step_duration
                                )
                                context.add_step(final_step)

                                # Track step metrics
                                step_metrics.append(
                                    StepMetric(
                                        step_id=final_step.step_id,
                                        step_type="generation",
                                        duration_ms=final_step_duration,
                                        iteration=current_iteration,
                                        tool_calls_count=0,
                                        generation_tokens=final_generation.usage.total_tokens,
                                        cache_hits=0,
                                        cache_misses=0,
                                    )
                                )

                                # Update context with final generation and complete execution
                                context.update_token_usage(final_generation.usage)
                                context.complete_execution()
                                total_tokens_processed += (
                                    final_generation.usage.total_tokens
                                )

                                # Calculate final metrics
                                total_execution_time = (
                                    time.perf_counter() - execution_start_time
                                ) * 1000
                                final_response_processing_time = 10.0

                                # Create performance metrics
                                performance_metrics = PerformanceMetrics(
                                    total_execution_time_ms=total_execution_time,
                                    input_processing_time_ms=input_processing_time,
                                    static_knowledge_processing_time_ms=static_knowledge_time,
                                    mcp_tools_preparation_time_ms=mcp_tools_time,
                                    generation_time_ms=generation_time_total,
                                    tool_execution_time_ms=tool_execution_time_total,
                                    final_response_processing_time_ms=final_response_processing_time,
                                    iteration_count=iteration_count,
                                    tool_calls_count=tool_calls_count,
                                    total_tokens_processed=total_tokens_processed,
                                    cache_hit_rate=(
                                        cache_hits / (cache_hits + cache_misses) * 100
                                    )
                                    if (cache_hits + cache_misses) > 0
                                    else 0.0,
                                    step_metrics=step_metrics,
                                    average_generation_time_ms=generation_time_total
                                    / max(
                                        1,
                                        len(
                                            [
                                                s
                                                for s in step_metrics
                                                if s.step_type == "generation"
                                            ]
                                        ),
                                    ),
                                    average_tool_execution_time_ms=tool_execution_time_total
                                    / max(1, tool_calls_count),
                                    longest_step_duration_ms=max(
                                        [s.duration_ms for s in step_metrics],
                                        default=0.0,
                                    ),
                                    shortest_step_duration_ms=min(
                                        [s.duration_ms for s in step_metrics],
                                        default=0.0,
                                    ),
                                )

                                # Save to conversation store after successful execution
                                if chat_id:
                                    assert self.conversation_store is not None
                                    await self.conversation_store.add_message_async(
                                        chat_id, current_user_message
                                    )
                                    await self.conversation_store.add_message_async(
                                        chat_id,
                                        final_generation.message.to_assistant_message(),
                                    )

                                # Yield final result
                                yield self._build_agent_run_output(
                                    context=context,
                                    generation=final_generation,
                                    performance_metrics=performance_metrics,
                                )
                                return

                        # If we got text and don't need structure, use what we have
                        _logger.bind_optional(
                            lambda log: log.debug(
                                "Using existing text response from streaming"
                            )
                        )

                        # Complete the final step and add to context
                        final_step_duration = (
                            time.perf_counter() - final_step_start_time
                        ) * 1000
                        final_step.mark_completed(duration_ms=final_step_duration)
                        context.add_step(final_step)

                        # Track step metrics
                        step_metrics.append(
                            StepMetric(
                                step_id=final_step.step_id,
                                step_type="generation",
                                duration_ms=final_step_duration,
                                iteration=current_iteration,
                                tool_calls_count=0,
                                generation_tokens=final_tool_generation.usage.total_tokens,
                                cache_hits=0,
                                cache_misses=0,
                            )
                        )

                        # Complete execution before returning
                        context.complete_execution()

                        # Calculate final metrics
                        total_execution_time = (
                            time.perf_counter() - execution_start_time
                        ) * 1000
                        final_response_processing_time = 5.0

                        # Create performance metrics
                        performance_metrics = PerformanceMetrics(
                            total_execution_time_ms=total_execution_time,
                            input_processing_time_ms=input_processing_time,
                            static_knowledge_processing_time_ms=static_knowledge_time,
                            mcp_tools_preparation_time_ms=mcp_tools_time,
                            generation_time_ms=generation_time_total,
                            tool_execution_time_ms=tool_execution_time_total,
                            final_response_processing_time_ms=final_response_processing_time,
                            iteration_count=iteration_count,
                            tool_calls_count=tool_calls_count,
                            total_tokens_processed=total_tokens_processed,
                            cache_hit_rate=(
                                cache_hits / (cache_hits + cache_misses) * 100
                            )
                            if (cache_hits + cache_misses) > 0
                            else 0.0,
                            step_metrics=step_metrics,
                            average_generation_time_ms=generation_time_total
                            / max(
                                1,
                                len(
                                    [
                                        s
                                        for s in step_metrics
                                        if s.step_type == "generation"
                                    ]
                                ),
                            ),
                            average_tool_execution_time_ms=tool_execution_time_total
                            / max(1, tool_calls_count),
                            longest_step_duration_ms=max(
                                [s.duration_ms for s in step_metrics], default=0.0
                            ),
                            shortest_step_duration_ms=min(
                                [s.duration_ms for s in step_metrics], default=0.0
                            ),
                        )

                        # Save to conversation store after successful execution
                        if chat_id:
                            assert self.conversation_store is not None
                            await self.conversation_store.add_message_async(
                                chat_id, current_user_message
                            )
                            await self.conversation_store.add_message_async(
                                chat_id,
                                final_tool_generation.message.to_assistant_message(),
                            )

                        # Yield final result
                        yield self._build_agent_run_output(
                            generation=cast(
                                Generation[T_Schema], final_tool_generation
                            ),
                            context=context,
                            performance_metrics=performance_metrics,
                        )
                        return

                    # FIXED: Process tools for this iteration only
                    _logger.bind_optional(
                        lambda log: log.info(
                            "Processing %d tool calls",
                            len(
                                cast(
                                    Generation[WithoutStructuredOutput],
                                    final_tool_generation,
                                ).tool_calls
                            ),
                        )
                    )

                    # FIXED: Create tool results for this iteration
                    tool_results_for_this_iteration: list[ToolExecutionResult] = []

                    # Create a step to track this iteration's tool executions
                    step_start_time = time.perf_counter()
                    step = Step(
                        step_type="tool_execution",
                        iteration=current_iteration,
                        tool_execution_suggestions=list(
                            final_tool_generation.tool_calls
                        ),
                        generation_text=final_tool_generation.text,
                        token_usage=final_tool_generation.usage,
                    )

                    # Track skipped tools for logging
                    skipped_tools: MutableSequence[str] = []
                    step_tool_execution_time = 0.0
                    step_tool_calls = 0

                    for tool_execution_suggestion in final_tool_generation.tool_calls:
                        pattern_key = self._get_tool_call_pattern_key(
                            tool_execution_suggestion.tool_name,
                            dict(tool_execution_suggestion.args),
                        )

                        # Check if this exact pattern was already called
                        pattern_count = tool_call_patterns.get(pattern_key, 0)

                        _logger.bind_optional(
                            lambda log: log.debug(
                                f"Tool pattern '{pattern_key}' has been called {pattern_count} times. "
                                + f"Max allowed: {self.agent_config.maxIdenticalToolCalls}"
                            )
                        )

                        if pattern_count >= self.agent_config.maxIdenticalToolCalls:
                            _logger.bind_optional(
                                lambda log: log.warning(
                                    "Blocking redundant tool call: %s with args: %s (already called %d times)",
                                    tool_execution_suggestion.tool_name,
                                    tool_execution_suggestion.args,
                                    pattern_count,
                                )
                            )

                            # Find the previous result from all_tool_results
                            previous_result = None
                            for (
                                prev_suggestion,
                                prev_result,
                            ) in all_tool_results.values():
                                if (
                                    prev_suggestion.tool_name
                                    == tool_execution_suggestion.tool_name
                                    and prev_suggestion.args
                                    == tool_execution_suggestion.args
                                ):
                                    previous_result = prev_result
                                    break

                            if previous_result is not None:
                                # FIXED: Create tool result for this iteration
                                tool_result = ToolExecutionResult(
                                    suggestion=tool_execution_suggestion,
                                    result=f"[DUPLICATE BLOCKED - Using previous result] {previous_result}",
                                    execution_time_ms=0.0,
                                    success=True,
                                    error_message=None,
                                )
                                tool_results_for_this_iteration.append(tool_result)

                                step.add_tool_execution_result(
                                    suggestion=tool_execution_suggestion,
                                    result=f"[DUPLICATE BLOCKED - Using previous result] {previous_result}",
                                    execution_time_ms=0,
                                    success=True,
                                )
                            else:
                                # This shouldn't happen, but handle it gracefully
                                error_msg = "Duplicate tool call blocked but no previous result found"
                                tool_result = ToolExecutionResult(
                                    suggestion=tool_execution_suggestion,
                                    result=error_msg,
                                    execution_time_ms=0.0,
                                    success=False,
                                    error_message=error_msg,
                                )
                                tool_results_for_this_iteration.append(tool_result)

                                step.add_tool_execution_result(
                                    suggestion=tool_execution_suggestion,
                                    result=error_msg,
                                    execution_time_ms=0,
                                    success=False,
                                    error_message=error_msg,
                                )

                            skipped_tools.append(
                                f"{tool_execution_suggestion.tool_name} (duplicate)"
                            )
                            continue

                        # Check budget
                        tool_name = tool_execution_suggestion.tool_name
                        current_tool_calls = tool_budget.get(tool_name, 0)
                        if current_tool_calls >= self.agent_config.maxCallPerTool:
                            _logger.bind_optional(
                                lambda log: log.warning(
                                    "Tool budget exceeded for: %s (already called %d times)",
                                    tool_name,
                                    current_tool_calls,
                                )
                            )

                            error_msg = f"Tool '{tool_name}' budget exceeded ({self.agent_config.maxCallPerTool} calls max)"
                            tool_result = ToolExecutionResult(
                                suggestion=tool_execution_suggestion,
                                result=error_msg,
                                execution_time_ms=0.0,
                                success=False,
                                error_message=error_msg,
                            )
                            tool_results_for_this_iteration.append(tool_result)

                            step.add_tool_execution_result(
                                suggestion=tool_execution_suggestion,
                                result=error_msg,
                                execution_time_ms=0,
                                success=False,
                                error_message=error_msg,
                            )

                            skipped_tools.append(f"{tool_name} (budget exceeded)")
                            continue

                        _logger.bind_optional(
                            lambda log: log.debug(
                                "Executing tool: %s with args: %s",
                                tool_execution_suggestion.tool_name,
                                tool_execution_suggestion.args,
                            )
                        )

                        selected_tool = available_tools[
                            tool_execution_suggestion.tool_name
                        ]

                        # Time the tool execution
                        tool_start_time = time.perf_counter()
                        try:
                            tool_result = await selected_tool.call_async(
                                **tool_execution_suggestion.args
                            )
                            tool_execution_time = (
                                time.perf_counter() - tool_start_time
                            ) * 1000

                            step_tool_execution_time += tool_execution_time
                            step_tool_calls += 1
                            tool_calls_count += 1
                            tool_execution_time_total += tool_execution_time

                            _logger.bind_optional(
                                lambda log: log.debug(
                                    "Tool execution result: %s", tool_result
                                )
                            )

                            # Update tracking structures
                            all_tool_results[tool_execution_suggestion.id] = (
                                tool_execution_suggestion,
                                tool_result,
                            )

                            # Update pattern tracking
                            tool_call_patterns[pattern_key] = pattern_count + 1

                            # Update budget
                            tool_budget[tool_name] = current_tool_calls + 1

                            # FIXED: Create tool result for this iteration
                            tool_execution_result = ToolExecutionResult(
                                suggestion=tool_execution_suggestion,
                                result=tool_result,
                                execution_time_ms=tool_execution_time,
                                success=True,
                                error_message=None,
                            )
                            tool_results_for_this_iteration.append(
                                tool_execution_result
                            )

                            # Add the tool execution result to the step
                            step.add_tool_execution_result(
                                suggestion=tool_execution_suggestion,
                                result=tool_result,
                                execution_time_ms=tool_execution_time,
                                success=True,
                            )
                        except ToolSuspensionError as suspension_error:
                            # Handle tool suspension for HITL workflows
                            suspension_reason = suspension_error.reason
                            _logger.bind_optional(
                                lambda log: log.info(
                                    "Tool execution suspended: %s", suspension_reason
                                )
                            )

                            # Get the suspension manager (injected or default)
                            suspension_mgr = (
                                self.suspension_manager
                                or get_default_suspension_manager()
                            )

                            # Save suspension state with tracking info
                            await self._save_suspension_state(
                                context=context,
                                suspension_type="tool_execution",
                                tool_suggestion=tool_execution_suggestion,
                                current_iteration=current_iteration,
                                all_tools=all_tools,
                                called_tools=all_tool_results,
                                current_step=step.model_dump()
                                if hasattr(step, "model_dump")
                                else None,
                            )

                            # Suspend the execution
                            resumption_token = await suspension_mgr.suspend_execution(
                                context=context,
                                reason=suspension_error.reason,
                                approval_data=suspension_error.approval_data,
                                timeout_hours=suspension_error.timeout_seconds // 3600
                                if suspension_error.timeout_seconds
                                else 24,
                            )

                            # Calculate partial metrics before suspension
                            total_execution_time = (
                                time.perf_counter() - execution_start_time
                            ) * 1000

                            # Create performance metrics for suspended execution
                            performance_metrics = PerformanceMetrics(
                                total_execution_time_ms=total_execution_time,
                                input_processing_time_ms=input_processing_time,
                                static_knowledge_processing_time_ms=static_knowledge_time,
                                mcp_tools_preparation_time_ms=mcp_tools_time,
                                generation_time_ms=generation_time_total,
                                tool_execution_time_ms=tool_execution_time_total,
                                final_response_processing_time_ms=0.0,
                                iteration_count=iteration_count,
                                tool_calls_count=tool_calls_count,
                                total_tokens_processed=total_tokens_processed,
                                cache_hit_rate=(
                                    cache_hits / (cache_hits + cache_misses) * 100
                                )
                                if (cache_hits + cache_misses) > 0
                                else 0.0,
                                step_metrics=step_metrics,
                                average_generation_time_ms=generation_time_total
                                / max(
                                    1,
                                    len(
                                        [
                                            s
                                            for s in step_metrics
                                            if s.step_type == "generation"
                                        ]
                                    ),
                                ),
                                average_tool_execution_time_ms=tool_execution_time_total
                                / max(1, tool_calls_count),
                                longest_step_duration_ms=max(
                                    [s.duration_ms for s in step_metrics], default=0.0
                                ),
                                shortest_step_duration_ms=min(
                                    [s.duration_ms for s in step_metrics], default=0.0
                                ),
                            )

                            # Yield suspended result
                            yield AgentRunOutput(
                                generation=None,
                                context=context,
                                parsed=cast(T_Schema, None),
                                is_suspended=True,
                                suspension_reason=suspension_error.reason,
                                resumption_token=resumption_token,
                                performance_metrics=performance_metrics,
                            )
                            return

                    # FIXED: Add tool results to context as a clean user message
                    if tool_results_for_this_iteration:
                        user_message_with_results = UserMessage(
                            parts=tool_results_for_this_iteration
                        )
                        context.message_history.append(user_message_with_results)

                    # Log skipped tools if any
                    if skipped_tools:
                        _logger.bind_optional(
                            lambda log: log.info(
                                "Skipped %d redundant tool calls: %s",
                                len(skipped_tools),
                                ", ".join(skipped_tools),
                            )
                        )

                    # Complete the step and add it to context
                    step_duration = (time.perf_counter() - step_start_time) * 1000
                    step.mark_completed(duration_ms=step_duration)
                    context.add_step(step)

                    # Track step metrics
                    step_metrics.append(
                        StepMetric(
                            step_id=step.step_id,
                            step_type="tool_execution",
                            duration_ms=step_duration,
                            iteration=current_iteration,
                            tool_calls_count=step_tool_calls,
                            generation_tokens=final_tool_generation.usage.total_tokens,
                            cache_hits=0,
                            cache_misses=0,
                        )
                    )

                    state.update(
                        last_response=final_tool_generation.text,
                        tool_calls_amount=final_tool_generation.tool_calls_amount(),
                        iteration=state.iteration + 1,
                        token_usage=final_tool_generation.usage,
                    )

                # If we reach here, we've exceeded max iterations
                execution_summary = self._format_tool_call_summary(all_tool_results)
                steps_summary = self._format_steps_summary(context.steps)
                pattern_analysis = self._analyze_tool_call_patterns(all_tool_results)

                execution_summary = dedent(f"""
                EXECUTION SUMMARY:
                ==================
                Iterations completed: {iteration_count}/{self.agent_config.maxIterations}
                Total messages in context: {len(context.message_history)}
                
                {execution_summary}
                
                {steps_summary}
                
                {pattern_analysis}
                """).strip()

                # Add token usage if available
                if context.total_token_usage:
                    usage = context.total_token_usage
                    execution_summary += f"\n\nToken usage: {usage.prompt_tokens} prompt + {usage.completion_tokens} completion = {usage.total_tokens} total"

                # Create enhanced error message
                enhanced_error_message = dedent(f"""
                Max tool calls exceeded after {self.agent_config.maxIterations} iterations.
                
                {execution_summary}
                
                RECOMMENDATIONS:
                - Review the tool calls above to identify potential loops or inefficiencies
                - Consider increasing maxIterations if the agent is making progress
                - Check if tools are returning expected results
                - Verify that the agent's instructions are clear and specific
                - Look for repeated calls with same arguments (potential infinite loops)
                """).strip()

                _logger.bind_optional(
                    lambda log: log.error(
                        "Max tool calls exceeded with execution summary:\n%s",
                        execution_summary,
                    )
                )

                # Mark context as failed due to max iterations exceeded
                context.fail_execution(error_message=enhanced_error_message)

                # Calculate final metrics before raising exception
                total_execution_time = (
                    time.perf_counter() - execution_start_time
                ) * 1000

                # Create performance metrics for failed execution
                performance_metrics = PerformanceMetrics(
                    total_execution_time_ms=total_execution_time,
                    input_processing_time_ms=input_processing_time,
                    static_knowledge_processing_time_ms=static_knowledge_time,
                    mcp_tools_preparation_time_ms=mcp_tools_time,
                    generation_time_ms=generation_time_total,
                    tool_execution_time_ms=tool_execution_time_total,
                    final_response_processing_time_ms=0.0,
                    iteration_count=iteration_count,
                    tool_calls_count=tool_calls_count,
                    total_tokens_processed=total_tokens_processed,
                    cache_hit_rate=(cache_hits / (cache_hits + cache_misses) * 100)
                    if (cache_hits + cache_misses) > 0
                    else 0.0,
                    step_metrics=step_metrics,
                    average_generation_time_ms=generation_time_total
                    / max(
                        1, len([s for s in step_metrics if s.step_type == "generation"])
                    ),
                    average_tool_execution_time_ms=tool_execution_time_total
                    / max(1, tool_calls_count),
                    longest_step_duration_ms=max(
                        [s.duration_ms for s in step_metrics], default=0.0
                    ),
                    shortest_step_duration_ms=min(
                        [s.duration_ms for s in step_metrics], default=0.0
                    ),
                )

                # Store metrics and execution summary in context for debugging
                context.metadata["performance_metrics"] = (
                    performance_metrics.model_dump()
                )
                context.metadata["execution_summary_on_failure"] = execution_summary

                raise MaxToolCallsExceededError(enhanced_error_message)

            return _stream_with_tools()

        # FIXED: Non-streaming path with tools - simplified message history management
        while state.iteration < self.agent_config.maxIterations:
            current_iteration = state.iteration + 1
            iteration_count = current_iteration
            _logger.bind_optional(
                lambda log: log.info(
                    "Starting iteration %d of %d",
                    current_iteration,
                    self.agent_config.maxIterations,
                )
            )

            # FIXED: Use context message history directly - no complex reconstruction
            message_history = context.message_history

            _logger.bind_optional(
                lambda log: log.debug("Generating tool call response")
            )

            generation_start = time.perf_counter()
            tool_call_generation = await generation_provider.generate_async(
                model=self.resolved_model,
                messages=message_history,
                generation_config=self.agent_config.generation_config,
                tools=all_tools,
            )

            generation_time_single = (time.perf_counter() - generation_start) * 1000
            generation_time_total += generation_time_single

            _logger.bind_optional(
                lambda log: log.debug(
                    "Tool call generation completed with %d tool calls",
                    tool_call_generation.tool_calls_amount(),
                )
            )

            # Update context with token usage from this generation
            context.update_token_usage(tool_call_generation.usage)
            total_tokens_processed += tool_call_generation.usage.total_tokens

            agent_didnt_call_any_tool = tool_call_generation.tool_calls_amount() == 0
            if agent_didnt_call_any_tool:
                _logger.bind_optional(
                    lambda log: log.info(
                        "Agent didn't call any tool, generating final response"
                    )
                )

                # Create a step for the final generation (no tools called)
                final_step_start_time = time.perf_counter()
                final_step = Step(
                    step_type="generation",
                    iteration=current_iteration,
                    tool_execution_suggestions=[],  # No tools called in this final step
                    generation_text=tool_call_generation.text
                    or "Generating final response...",
                    token_usage=tool_call_generation.usage,
                )

                # Only make another call if we need structured output or didn't get text
                if self.response_schema is not None or not tool_call_generation.text:
                    _logger.bind_optional(
                        lambda log: log.debug("Generating structured response")
                    )

                    # Generate final structured response
                    generation_start = time.perf_counter()
                    generation = await generation_provider.generate_async(
                        model=self.resolved_model,
                        messages=list(context.message_history)
                        + [tool_call_generation.message.to_assistant_message()],
                        response_schema=self.response_schema,
                        generation_config=self.agent_config.generation_config,
                    )

                    context.message_history.append(
                        generation.message.to_assistant_message()
                    )
                    generation_time_single = (
                        time.perf_counter() - generation_start
                    ) * 1000
                    generation_time_total += generation_time_single

                    _logger.bind_optional(
                        lambda log: log.debug("Final generation complete")
                    )

                    # Update the step with the final generation results
                    final_step.generation_text = generation.text
                    final_step.token_usage = generation.usage
                    final_step_duration = (
                        time.perf_counter() - final_step_start_time
                    ) * 1000
                    final_step.mark_completed(duration_ms=final_step_duration)
                    context.add_step(final_step)

                    # Track step metrics
                    step_metrics.append(
                        StepMetric(
                            step_id=final_step.step_id,
                            step_type="generation",
                            duration_ms=final_step_duration,
                            iteration=current_iteration,
                            tool_calls_count=0,
                            generation_tokens=generation.usage.total_tokens,
                            cache_hits=0,
                            cache_misses=0,
                        )
                    )

                    # Update context with final generation and complete execution
                    context.update_token_usage(generation.usage)
                    context.complete_execution()
                    total_tokens_processed += generation.usage.total_tokens

                    # Calculate final metrics
                    total_execution_time = (
                        time.perf_counter() - execution_start_time
                    ) * 1000
                    final_response_processing_time = 10.0

                    # Create performance metrics
                    performance_metrics = PerformanceMetrics(
                        total_execution_time_ms=total_execution_time,
                        input_processing_time_ms=input_processing_time,
                        static_knowledge_processing_time_ms=static_knowledge_time,
                        mcp_tools_preparation_time_ms=mcp_tools_time,
                        generation_time_ms=generation_time_total,
                        tool_execution_time_ms=tool_execution_time_total,
                        final_response_processing_time_ms=final_response_processing_time,
                        iteration_count=iteration_count,
                        tool_calls_count=tool_calls_count,
                        total_tokens_processed=total_tokens_processed,
                        cache_hit_rate=(cache_hits / (cache_hits + cache_misses) * 100)
                        if (cache_hits + cache_misses) > 0
                        else 0.0,
                        step_metrics=step_metrics,
                        average_generation_time_ms=generation_time_total
                        / max(
                            1,
                            len(
                                [s for s in step_metrics if s.step_type == "generation"]
                            ),
                        ),
                        average_tool_execution_time_ms=tool_execution_time_total
                        / max(1, tool_calls_count),
                        longest_step_duration_ms=max(
                            [s.duration_ms for s in step_metrics], default=0.0
                        ),
                        shortest_step_duration_ms=min(
                            [s.duration_ms for s in step_metrics], default=0.0
                        ),
                    )

                    # Save to conversation store after successful execution
                    if chat_id:
                        assert self.conversation_store is not None
                        # Add the current user message and AI response to conversation store
                        await self.conversation_store.add_message_async(
                            chat_id, current_user_message
                        )
                        await self.conversation_store.add_message_async(
                            chat_id, generation.message.to_assistant_message()
                        )

                    return self._build_agent_run_output(
                        context=context,
                        generation=generation,
                        performance_metrics=performance_metrics,
                    )

                # If we got text and don't need structure, use what we have
                _logger.bind_optional(
                    lambda log: log.debug("Using existing text response")
                )

                # Complete the final step and add to context
                final_step_duration = (
                    time.perf_counter() - final_step_start_time
                ) * 1000
                final_step.mark_completed(duration_ms=final_step_duration)
                context.add_step(final_step)

                # Track step metrics
                step_metrics.append(
                    StepMetric(
                        step_id=final_step.step_id,
                        step_type="generation",
                        duration_ms=final_step_duration,
                        iteration=current_iteration,
                        tool_calls_count=0,
                        generation_tokens=tool_call_generation.usage.total_tokens,
                        cache_hits=0,
                        cache_misses=0,
                    )
                )

                # Complete execution before returning
                context.complete_execution()

                # Calculate final metrics
                total_execution_time = (
                    time.perf_counter() - execution_start_time
                ) * 1000
                final_response_processing_time = 5.0

                # Create performance metrics
                performance_metrics = PerformanceMetrics(
                    total_execution_time_ms=total_execution_time,
                    input_processing_time_ms=input_processing_time,
                    static_knowledge_processing_time_ms=static_knowledge_time,
                    mcp_tools_preparation_time_ms=mcp_tools_time,
                    generation_time_ms=generation_time_total,
                    tool_execution_time_ms=tool_execution_time_total,
                    final_response_processing_time_ms=final_response_processing_time,
                    iteration_count=iteration_count,
                    tool_calls_count=tool_calls_count,
                    total_tokens_processed=total_tokens_processed,
                    cache_hit_rate=(cache_hits / (cache_hits + cache_misses) * 100)
                    if (cache_hits + cache_misses) > 0
                    else 0.0,
                    step_metrics=step_metrics,
                    average_generation_time_ms=generation_time_total
                    / max(
                        1, len([s for s in step_metrics if s.step_type == "generation"])
                    ),
                    average_tool_execution_time_ms=tool_execution_time_total
                    / max(1, tool_calls_count),
                    longest_step_duration_ms=max(
                        [s.duration_ms for s in step_metrics], default=0.0
                    ),
                    shortest_step_duration_ms=min(
                        [s.duration_ms for s in step_metrics], default=0.0
                    ),
                )

                # Save to conversation store after successful execution
                if chat_id:
                    assert self.conversation_store is not None
                    # Add the current user message and AI response to conversation store
                    await self.conversation_store.add_message_async(
                        chat_id, current_user_message
                    )
                    await self.conversation_store.add_message_async(
                        chat_id, tool_call_generation.message.to_assistant_message()
                    )

                return self._build_agent_run_output(
                    generation=cast(Generation[T_Schema], tool_call_generation),
                    context=context,
                    performance_metrics=performance_metrics,
                )

            # FIXED: Process tools and create clean tool results for this iteration
            _logger.bind_optional(
                lambda log: log.info(
                    "Processing %d tool calls", len(tool_call_generation.tool_calls)
                )
            )

            # FIXED: Create tool results for this iteration only
            tool_results_for_this_iteration: list[ToolExecutionResult] = []

            # Create a step to track this iteration's tool executions
            step_start_time = time.perf_counter()
            step = Step(
                step_type="tool_execution",
                iteration=current_iteration,
                tool_execution_suggestions=list(tool_call_generation.tool_calls),
                generation_text=tool_call_generation.text,
                token_usage=tool_call_generation.usage,
            )

            # Track skipped tools for logging
            skipped_tools: MutableSequence[str] = []
            step_tool_execution_time = 0.0
            step_tool_calls = 0

            for tool_execution_suggestion in tool_call_generation.tool_calls:
                pattern_key = self._get_tool_call_pattern_key(
                    tool_execution_suggestion.tool_name,
                    dict(tool_execution_suggestion.args),
                )

                # Check if this exact pattern was already called
                pattern_count = tool_call_patterns.get(pattern_key, 0)

                _logger.bind_optional(
                    lambda log: log.debug(
                        f"Tool pattern '{pattern_key}' has been called {pattern_count} times. "
                        + f"Max allowed: {self.agent_config.maxIdenticalToolCalls}"
                    )
                )

                if pattern_count >= self.agent_config.maxIdenticalToolCalls:
                    _logger.bind_optional(
                        lambda log: log.warning(
                            "Blocking redundant tool call: %s with args: %s (already called %d times)",
                            tool_execution_suggestion.tool_name,
                            tool_execution_suggestion.args,
                            pattern_count,
                        )
                    )

                    # Find the previous result from all_tool_results
                    previous_result = None
                    for prev_suggestion, prev_result in all_tool_results.values():
                        if (
                            prev_suggestion.tool_name
                            == tool_execution_suggestion.tool_name
                            and prev_suggestion.args == tool_execution_suggestion.args
                        ):
                            previous_result = prev_result
                            break

                    if previous_result is not None:
                        # FIXED: Create tool result for this iteration
                        tool_result = ToolExecutionResult(
                            suggestion=tool_execution_suggestion,
                            result=f"[DUPLICATE BLOCKED - Using previous result] {previous_result}",
                            execution_time_ms=0.0,
                            success=True,
                            error_message=None,
                        )
                        tool_results_for_this_iteration.append(tool_result)

                        step.add_tool_execution_result(
                            suggestion=tool_execution_suggestion,
                            result=f"[DUPLICATE BLOCKED - Using previous result] {previous_result}",
                            execution_time_ms=0,
                            success=True,
                        )
                    else:
                        # This shouldn't happen, but handle it gracefully
                        error_msg = (
                            "Duplicate tool call blocked but no previous result found"
                        )
                        tool_result = ToolExecutionResult(
                            suggestion=tool_execution_suggestion,
                            result=error_msg,
                            execution_time_ms=0.0,
                            success=False,
                            error_message=error_msg,
                        )
                        tool_results_for_this_iteration.append(tool_result)

                        step.add_tool_execution_result(
                            suggestion=tool_execution_suggestion,
                            result=error_msg,
                            execution_time_ms=0,
                            success=False,
                            error_message=error_msg,
                        )

                    skipped_tools.append(
                        f"{tool_execution_suggestion.tool_name} (duplicate)"
                    )
                    continue

                # Check budget
                tool_name = tool_execution_suggestion.tool_name
                current_tool_calls = tool_budget.get(tool_name, 0)
                if current_tool_calls >= self.agent_config.maxCallPerTool:
                    _logger.bind_optional(
                        lambda log: log.warning(
                            "Tool budget exceeded for: %s (already called %d times)",
                            tool_name,
                            current_tool_calls,
                        )
                    )

                    error_msg = f"Tool '{tool_name}' budget exceeded ({self.agent_config.maxCallPerTool} calls max)"
                    tool_result = ToolExecutionResult(
                        suggestion=tool_execution_suggestion,
                        result=error_msg,
                        execution_time_ms=0.0,
                        success=False,
                        error_message=error_msg,
                    )
                    tool_results_for_this_iteration.append(tool_result)

                    step.add_tool_execution_result(
                        suggestion=tool_execution_suggestion,
                        result=error_msg,
                        execution_time_ms=0,
                        success=False,
                        error_message=error_msg,
                    )

                    skipped_tools.append(f"{tool_name} (budget exceeded)")
                    continue

                _logger.bind_optional(
                    lambda log: log.debug(
                        "Executing tool: %s with args: %s",
                        tool_execution_suggestion.tool_name,
                        tool_execution_suggestion.args,
                    )
                )

                selected_tool = available_tools[tool_execution_suggestion.tool_name]

                # Time the tool execution
                tool_start_time = time.perf_counter()
                try:
                    tool_result = await selected_tool.call_async(
                        **tool_execution_suggestion.args
                    )
                    tool_execution_time = (
                        time.perf_counter() - tool_start_time
                    ) * 1000  # Convert to milliseconds

                    step_tool_execution_time += tool_execution_time
                    step_tool_calls += 1
                    tool_calls_count += 1
                    tool_execution_time_total += tool_execution_time

                    _logger.bind_optional(
                        lambda log: log.debug("Tool execution result: %s", tool_result)
                    )

                    # Update tracking structures
                    all_tool_results[tool_execution_suggestion.id] = (
                        tool_execution_suggestion,
                        tool_result,
                    )

                    # Update pattern tracking
                    tool_call_patterns[pattern_key] = pattern_count + 1

                    # Update budget
                    tool_budget[tool_name] = current_tool_calls + 1

                    # FIXED: Create tool result for this iteration
                    tool_execution_result = ToolExecutionResult(
                        suggestion=tool_execution_suggestion,
                        result=tool_result,
                        execution_time_ms=tool_execution_time,
                        success=True,
                        error_message=None,
                    )
                    tool_results_for_this_iteration.append(tool_execution_result)

                    # Add the tool execution result to the step
                    step.add_tool_execution_result(
                        suggestion=tool_execution_suggestion,
                        result=tool_result,
                        execution_time_ms=tool_execution_time,
                        success=True,
                    )
                except ToolSuspensionError as suspension_error:
                    # Handle tool suspension for HITL workflows
                    suspension_reason = suspension_error.reason
                    _logger.bind_optional(
                        lambda log: log.info(
                            "Tool execution suspended: %s", suspension_reason
                        )
                    )

                    # Get the suspension manager (injected or default)
                    suspension_mgr = (
                        self.suspension_manager or get_default_suspension_manager()
                    )

                    # Save suspension state with tracking info
                    await self._save_suspension_state(
                        context=context,
                        suspension_type="tool_execution",
                        tool_suggestion=tool_execution_suggestion,
                        current_iteration=current_iteration,
                        all_tools=all_tools,
                        called_tools=all_tool_results,
                        current_step=step.model_dump()
                        if hasattr(step, "model_dump")
                        else None,
                    )

                    # Suspend the execution
                    resumption_token = await suspension_mgr.suspend_execution(
                        context=context,
                        reason=suspension_error.reason,
                        approval_data=suspension_error.approval_data,
                        timeout_hours=suspension_error.timeout_seconds // 3600
                        if suspension_error.timeout_seconds
                        else 24,
                    )

                    # Calculate partial metrics before suspension
                    total_execution_time = (
                        time.perf_counter() - execution_start_time
                    ) * 1000

                    # Create performance metrics for suspended execution
                    performance_metrics = PerformanceMetrics(
                        total_execution_time_ms=total_execution_time,
                        input_processing_time_ms=input_processing_time,
                        static_knowledge_processing_time_ms=static_knowledge_time,
                        mcp_tools_preparation_time_ms=mcp_tools_time,
                        generation_time_ms=generation_time_total,
                        tool_execution_time_ms=tool_execution_time_total,
                        final_response_processing_time_ms=0.0,  # Not completed yet
                        iteration_count=iteration_count,
                        tool_calls_count=tool_calls_count,
                        total_tokens_processed=total_tokens_processed,
                        cache_hit_rate=(cache_hits / (cache_hits + cache_misses) * 100)
                        if (cache_hits + cache_misses) > 0
                        else 0.0,
                        step_metrics=step_metrics,
                        average_generation_time_ms=generation_time_total
                        / max(
                            1,
                            len(
                                [s for s in step_metrics if s.step_type == "generation"]
                            ),
                        ),
                        average_tool_execution_time_ms=tool_execution_time_total
                        / max(1, tool_calls_count),
                        longest_step_duration_ms=max(
                            [s.duration_ms for s in step_metrics], default=0.0
                        ),
                        shortest_step_duration_ms=min(
                            [s.duration_ms for s in step_metrics], default=0.0
                        ),
                    )

                    # Return suspended result immediately
                    return AgentRunOutput(
                        generation=None,
                        context=context,
                        parsed=cast(T_Schema, None),
                        is_suspended=True,
                        suspension_reason=suspension_error.reason,
                        resumption_token=resumption_token,
                        performance_metrics=performance_metrics,
                    )

            # FIXED: Add tool results to context as a clean user message
            if tool_results_for_this_iteration:
                user_message_with_results = UserMessage(
                    parts=tool_results_for_this_iteration
                )
                context.message_history.append(user_message_with_results)

            # Log skipped tools if any
            if skipped_tools:
                _logger.bind_optional(
                    lambda log: log.info(
                        "Skipped %d redundant tool calls: %s",
                        len(skipped_tools),
                        ", ".join(skipped_tools),
                    )
                )

            # Complete the step and add it to context
            step_duration = (
                time.perf_counter() - step_start_time
            ) * 1000  # Convert to milliseconds
            step.mark_completed(duration_ms=step_duration)
            context.add_step(step)

            # Track step metrics
            step_metrics.append(
                StepMetric(
                    step_id=step.step_id,
                    step_type="tool_execution",
                    duration_ms=step_duration,
                    iteration=current_iteration,
                    tool_calls_count=step_tool_calls,
                    generation_tokens=tool_call_generation.usage.total_tokens,
                    cache_hits=0,
                    cache_misses=0,
                )
            )

            state.update(
                last_response=tool_call_generation.text,
                tool_calls_amount=tool_call_generation.tool_calls_amount(),
                iteration=state.iteration + 1,
                token_usage=tool_call_generation.usage,
            )

        # Generate detailed execution summary
        tool_call_summary = self._format_tool_call_summary(all_tool_results)
        steps_summary = self._format_steps_summary(context.steps)
        pattern_analysis = self._analyze_tool_call_patterns(all_tool_results)

        execution_summary = dedent(f"""
        EXECUTION SUMMARY:
        ==================
        Iterations completed: {iteration_count}/{self.agent_config.maxIterations}
        Total messages in context: {len(context.message_history)}

        {tool_call_summary}

        {steps_summary}

        {pattern_analysis}
        """).strip()

        # Add token usage if available
        if context.total_token_usage:
            usage = context.total_token_usage
            execution_summary += f"\n\nToken usage: {usage.prompt_tokens} prompt + {usage.completion_tokens} completion = {usage.total_tokens} total"

        # Create enhanced error message
        enhanced_error_message = dedent(f"""
        Max tool calls exceeded after {self.agent_config.maxIterations} iterations.

        {execution_summary}

        RECOMMENDATIONS:
        - Review the tool calls above to identify potential loops or inefficiencies
        - Consider increasing maxIterations if the agent is making progress
        - Check if tools are returning expected results
        - Verify that the agent's instructions are clear and specific
        - Look for repeated calls with same arguments (potential infinite loops)
        """).strip()

        _logger.bind_optional(
            lambda log: log.error(
                "Max tool calls exceeded with execution summary:\n%s", execution_summary
            )
        )

        # Mark context as failed due to max iterations exceeded
        context.fail_execution(error_message=enhanced_error_message)

        # Calculate final metrics before raising exception
        total_execution_time = (time.perf_counter() - execution_start_time) * 1000

        # Create performance metrics for failed execution
        performance_metrics = PerformanceMetrics(
            total_execution_time_ms=total_execution_time,
            input_processing_time_ms=input_processing_time,
            static_knowledge_processing_time_ms=static_knowledge_time,
            mcp_tools_preparation_time_ms=mcp_tools_time,
            generation_time_ms=generation_time_total,
            tool_execution_time_ms=tool_execution_time_total,
            final_response_processing_time_ms=0.0,  # Failed before completion
            iteration_count=iteration_count,
            tool_calls_count=tool_calls_count,
            total_tokens_processed=total_tokens_processed,
            cache_hit_rate=(cache_hits / (cache_hits + cache_misses) * 100)
            if (cache_hits + cache_misses) > 0
            else 0.0,
            step_metrics=step_metrics,
            average_generation_time_ms=generation_time_total
            / max(1, len([s for s in step_metrics if s.step_type == "generation"])),
            average_tool_execution_time_ms=tool_execution_time_total
            / max(1, tool_calls_count),
            longest_step_duration_ms=max(
                [s.duration_ms for s in step_metrics], default=0.0
            ),
            shortest_step_duration_ms=min(
                [s.duration_ms for s in step_metrics], default=0.0
            ),
        )

        # Store metrics and execution summary in context for debugging
        context.metadata["performance_metrics"] = performance_metrics.model_dump()
        context.metadata["execution_summary_on_failure"] = execution_summary

        raise MaxToolCallsExceededError(enhanced_error_message)

    def to_api(self, *extra_routes: type[Controller]) -> Application:
        from agentle.agents.asgi.blacksheep.agent_to_blacksheep_application_adapter import (
            AgentToBlackSheepApplicationAdapter,
        )

        return AgentToBlackSheepApplicationAdapter(*extra_routes).adapt(self)

    def clone(
        self,
        *,
        new_name: str | None = None,
        new_instructions: str | None = None,
        new_tools: MutableSequence[
            Tool | Callable[..., Any] | Callable[..., Awaitable[Any]]
        ]
        | None = None,
        new_config: AgentConfig | AgentConfigDict | None = None,
        new_model: str | None = None,
        new_version: str | None = None,
        new_documentation_url: str | None = None,
        new_capabilities: Capabilities | None = None,
        new_authentication: Authentication | None = None,
        new_default_input_modes: Sequence[str] | None = None,
        new_default_output_modes: Sequence[str] | None = None,
        new_skills: Sequence[AgentSkill] | None = None,
        new_mcp_servers: MutableSequence[MCPServerProtocol] | None = None,
        new_generation_provider: GenerationProviderType | None = None,
        new_url: str | None = None,
        new_suspension_manager: SuspensionManager | None = None,
        new_document_cache_store: DocumentCacheStoreType | None = None,
    ) -> Agent[T_Schema]:
        """
        Creates a clone of the current agent with optionally modified attributes.

        This method facilitates creating variations of an agent without modifying the original.
        Unspecified parameters will retain the values from the original agent.

        Args:
            new_name: New name for the agent.
            new_instructions: New instructions for the agent.
            new_tools: New tools for the agent.
            new_config: New configuration for the agent.
            new_model: New model for the agent.
            new_version: New version for the agent.
            new_documentation_url: New documentation URL for the agent.
            new_capabilities: New capabilities for the agent.
            new_authentication: New authentication for the agent.
            new_default_input_modes: New default input modes for the agent.
            new_default_output_modes: New default output modes for the agent.
            new_skills: New skills for the agent.
            new_mcp_servers: New MCP servers for the agent.
            new_generation_provider: New generation provider for the agent.
            new_url: New URL for the agent.
            new_suspension_manager: New suspension manager for the agent.
            new_document_cache_store: New cache store for the agent.

        Returns:
            Agent[T_Schema]: A new agent with the specified attributes modified.

        Example:
            ```python
            # Create a variation of the agent with different instructions
            weather_agent_fr = weather_agent.clone(
                new_name="French Weather Agent",
                new_instructions="You are a weather agent that can answer questions about the weather in French."
            )
            ```
        """
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
            suspension_manager=new_suspension_manager or self.suspension_manager,
            document_cache_store=new_document_cache_store or self.document_cache_store,
        )

    def to_streamlit(
        self,
        *,
        title: str = "AI Agent",
        description: str = "No description.",
        initial_mode: Literal["dev", "presentation"] = "presentation",
    ) -> Callable[[], None]:
        return AgentToStreamlit[T_Schema](
            title=title,
            description=description,
            initial_mode=initial_mode,
        ).adapt(self)

    async def _continue_execution_from_context(
        self, context: Context
    ) -> AgentRunOutput[T_Schema]:
        """
        Continue agent execution from a resumed context.

        This method is used internally to resume execution after a suspension.
        It continues from where the agent left off based on the context state.

        The method handles various suspension scenarios:
        1. Tool execution suspension (most common)
        2. Generation suspension (less common)
        3. Complex pipeline/team suspension scenarios

        It properly restores execution state and continues from the exact
        suspension point, ensuring no work is lost or duplicated.
        """
        _logger = Maybe(logger if self.debug else None)

        _logger.bind_optional(
            lambda log: log.info(
                "Resuming agent execution from suspended context: %s",
                context.context_id,
            )
        )

        # Check if there's approval data in the context
        approval_result = context.get_checkpoint_data("approval_result")

        # Handle approval denial
        if approval_result and not approval_result.get("approved", True):
            reason = approval_result.get("approval_data", {}).get(
                "reason", "No reason provided"
            )
            denial_message = f"Request denied by {approval_result.get('approver_id', 'unknown')}: {reason}"

            _logger.bind_optional(
                lambda log: log.info(
                    "Execution denied during resumption: %s", denial_message
                )
            )

            context.fail_execution(denial_message)
            return AgentRunOutput(
                generation=None,
                context=context,
                parsed=cast(T_Schema, None),
                is_suspended=False,
                suspension_reason=denial_message,
            )

        try:
            # Resume the context execution state
            context.resume_execution()

            # Get suspension state data
            suspension_state = context.get_checkpoint_data("suspension_state")

            if not suspension_state:
                # If no suspension state, this might be a legacy suspension
                # Complete execution and return
                _logger.bind_optional(
                    lambda log: log.warning(
                        "No suspension state found, completing execution"
                    )
                )
                context.complete_execution()
                return AgentRunOutput(
                    generation=None,
                    context=context,
                    parsed=cast(T_Schema, None),
                    is_suspended=False,
                )

            suspension_type = suspension_state.get("type", "unknown")
            _logger.bind_optional(
                lambda log: log.debug(
                    "Resuming from suspension type: %s", suspension_type
                )
            )

            if suspension_type == "tool_execution":
                return await self._resume_from_tool_suspension(
                    context, suspension_state
                )
            elif suspension_type == "generation":
                return await self._resume_from_generation_suspension(
                    context, suspension_state
                )
            else:
                # Unknown suspension type, try to continue with normal flow
                _logger.bind_optional(
                    lambda log: log.warning(
                        "Unknown suspension type: %s, attempting normal continuation",
                        suspension_type,
                    )
                )
                return await self._resume_with_normal_flow(context)

        except Exception as e:
            error_message = f"Error during execution resumption: {str(e)}"
            _logger.bind_optional(
                lambda log: log.error(
                    "Error during execution resumption: %s", error_message
                )
            )
            context.fail_execution(error_message)
            return AgentRunOutput(
                generation=None,
                context=context,
                parsed=cast(T_Schema, None),
                is_suspended=False,
                suspension_reason=error_message,
            )

    async def _resume_from_tool_suspension(
        self, context: Context, suspension_state: dict[str, Any]
    ) -> AgentRunOutput[T_Schema]:
        """
        Resume execution from a tool suspension point.

        This handles the most common suspension scenario where a tool
        raised ToolSuspensionError and required approval.
        """
        _logger = Maybe(logger if self.debug else None)

        # Extract suspension state
        suspended_tool_suggestion = suspension_state.get("tool_suggestion")
        current_iteration = suspension_state.get("current_iteration", 1)
        called_tools = suspension_state.get("called_tools", {})
        current_step = suspension_state.get("current_step")

        if not suspended_tool_suggestion:
            raise ValueError("No suspended tool suggestion found in suspension state")

        _logger.bind_optional(
            lambda log: log.debug(
                "Resuming tool execution: %s",
                suspended_tool_suggestion.get("tool_name"),
            )
        )

        all_tools: Sequence[Tool] = cast(Sequence[Tool], await self._all_tools())

        available_tools: MutableMapping[str, Tool[Any]] = {
            tool.name: tool for tool in all_tools
        }

        # Create ToolExecutionSuggestion from saved data
        tool_suggestion = ToolExecutionSuggestion(
            id=suspended_tool_suggestion["id"],
            tool_name=suspended_tool_suggestion["tool_name"],
            args=suspended_tool_suggestion["args"],
        )

        # Approval data is stored in context for tools that need it

        # Reconstruct or create the current step
        if current_step:
            step = Step(
                step_id=current_step["step_id"],
                step_type=current_step["step_type"],
                iteration=current_step["iteration"],
                tool_execution_suggestions=current_step["tool_execution_suggestions"],
                generation_text=current_step.get("generation_text"),
                token_usage=current_step.get("token_usage"),
            )
            # Update step timestamp to reflect resumption
            step.timestamp = datetime.datetime.now()
        else:
            # Create new step for the resumed execution
            step = Step(
                step_type="tool_execution",
                iteration=current_iteration,
                tool_execution_suggestions=[tool_suggestion],
                generation_text="Resuming from suspension...",
            )

        step_start_time = time.time()

        # Execute the approved tool
        selected_tool = available_tools.get(tool_suggestion.tool_name)
        if not selected_tool:
            raise ValueError(
                f"Tool '{tool_suggestion.tool_name}' not found in available tools"
            )

        _logger.bind_optional(
            lambda log: log.debug(
                "Executing approved tool: %s with args: %s",
                tool_suggestion.tool_name,
                tool_suggestion.args,
            )
        )

        # Use the tool arguments from the suspended suggestion
        tool_args = dict(tool_suggestion.args)
        # Note: Approval data is available in context checkpoint data if tools need it

        # Execute the tool
        tool_start_time = time.time()
        try:
            tool_result = await selected_tool.call_async(**tool_args)
            tool_execution_time = (time.time() - tool_start_time) * 1000

            _logger.bind_optional(
                lambda log: log.debug(
                    "Tool execution completed successfully: %s", str(tool_result)[:100]
                )
            )

            # Add the successful tool execution to the step
            step.add_tool_execution_result(
                suggestion=tool_suggestion,
                result=tool_result,
                execution_time_ms=tool_execution_time,
                success=True,
            )

            # Update called_tools with the result
            called_tools[tool_suggestion.id] = (tool_suggestion, tool_result)

        except ToolSuspensionError as suspension_error:
            # The tool suspended again - handle nested suspension
            error = suspension_error

            _logger.bind_optional(
                lambda log: log.info(
                    "Tool suspended again during resumption: %s",
                    error.reason,
                )
            )

            # Save the current state for the new suspension
            await self._save_suspension_state(
                context=context,
                suspension_type="tool_execution",
                tool_suggestion=tool_suggestion,
                current_iteration=current_iteration,
                all_tools=all_tools,
                called_tools=called_tools,
                current_step=step.model_dump() if hasattr(step, "model_dump") else None,
            )

            # Get suspension manager and suspend again
            suspension_mgr = self.suspension_manager or get_default_suspension_manager()
            resumption_token = await suspension_mgr.suspend_execution(
                context=context,
                reason=suspension_error.reason,
                approval_data=suspension_error.approval_data,
                timeout_hours=suspension_error.timeout_seconds // 3600
                if suspension_error.timeout_seconds
                else 24,
            )

            return AgentRunOutput(
                generation=None,
                context=context,
                parsed=cast(T_Schema, None),
                is_suspended=True,
                suspension_reason=suspension_error.reason,
                resumption_token=resumption_token,
            )
        except Exception as e:
            # Tool execution failed
            tool_execution_time = (time.time() - tool_start_time) * 1000
            error_message = f"Tool execution failed: {str(e)}"

            _logger.bind_optional(
                lambda log: log.error(
                    "Tool execution failed during resumption: %s", error_message
                )
            )

            step.add_tool_execution_result(
                suggestion=tool_suggestion,
                result=error_message,
                execution_time_ms=tool_execution_time,
                success=False,
                error_message=error_message,
            )

            # Complete the step and add to context
            step_duration = (time.time() - step_start_time) * 1000
            step.mark_failed(error_message=error_message, duration_ms=step_duration)
            context.add_step(step)

            # Fail the execution
            context.fail_execution(error_message)
            return AgentRunOutput(
                generation=None,
                context=context,
                parsed=cast(T_Schema, None),
                is_suspended=False,
                suspension_reason=error_message,
            )

        # Complete the step and add to context
        step_duration = (time.time() - step_start_time) * 1000
        step.mark_completed(duration_ms=step_duration)
        context.add_step(step)

        # Clear the suspension state since we've handled it
        context.set_checkpoint_data("suspension_state", None)

        # Continue with the normal agent execution flow
        _logger.bind_optional(
            lambda log: log.debug(
                "Tool execution completed, continuing with normal flow"
            )
        )

        return await self._continue_normal_execution_flow(
            context=context,
            current_iteration=current_iteration,
            all_tools=all_tools,
            called_tools=called_tools,
        )

    async def _resume_from_generation_suspension(
        self, context: Context, suspension_state: dict[str, Any]
    ) -> AgentRunOutput[T_Schema]:
        """
        Resume execution from a generation suspension point.

        This handles cases where the generation itself was suspended
        (less common but possible in some scenarios).
        """
        _logger = Maybe(logger if self.debug else None)

        _logger.bind_optional(
            lambda log: log.debug("Resuming from generation suspension")
        )

        # For generation suspension, we typically just continue with normal flow
        # The approval has already been processed by this point
        context.set_checkpoint_data("suspension_state", None)
        return await self._resume_with_normal_flow(context)

    async def _resume_with_normal_flow(
        self, context: Context
    ) -> AgentRunOutput[T_Schema]:
        """
        Resume with normal agent execution flow.

        This is used when we can't determine the exact suspension point
        or for simple resumption scenarios.
        """
        _logger = Maybe(logger if self.debug else None)
        generation_provider = self.generation_provider

        _logger.bind_optional(
            lambda log: log.debug("Resuming with normal execution flow")
        )

        # Check if agent has tools
        mcp_tools: MutableSequence[tuple[MCPServerProtocol, MCPTool]] = []
        if self.mcp_servers:
            for server in self.mcp_servers:
                tools = await server.list_tools_async()
                mcp_tools.extend((server, tool) for tool in tools)

        agent_has_tools = self.has_tools() or len(mcp_tools) > 0

        if not agent_has_tools:
            # No tools, generate final response
            generation = await generation_provider.generate_async(
                model=self.resolved_model,
                messages=context.message_history,
                response_schema=self.response_schema,
                generation_config=self.agent_config.generation_config,
            )

            context.update_token_usage(generation.usage)
            context.complete_execution()

            return AgentRunOutput(
                generation=generation,
                context=context,
                parsed=generation.parsed,
            )

        # Has tools, continue with tool execution loop
        all_tools: MutableSequence[Tool[Any]] = [
            Tool.from_mcp_tool(mcp_tool=tool, server=server)
            for server, tool in mcp_tools
        ] + [
            Tool.from_callable(tool) if callable(tool) else tool for tool in self.tools
        ]

        return await self._continue_normal_execution_flow(
            context=context,
            current_iteration=context.execution_state.current_iteration + 1,
            all_tools=all_tools,
            called_tools={},
        )

    async def _continue_normal_execution_flow(
        self,
        context: Context,
        current_iteration: int,
        all_tools: Sequence[Tool[Any]],
        called_tools: dict[str, tuple[ToolExecutionSuggestion, Any]],
    ) -> AgentRunOutput[T_Schema]:
        """
        Continue the normal agent execution flow after resumption.

        This method continues the standard tool execution loop from where
        the agent left off, handling iterations and tool calls.
        """
        _logger = Maybe(logger if self.debug else None)
        generation_provider = self.generation_provider

        available_tools: MutableMapping[str, Tool[Any]] = {
            tool.name: tool for tool in all_tools
        }

        # Continue the execution loop from current iteration
        while current_iteration <= self.agent_config.maxIterations:
            _logger.bind_optional(
                lambda log: log.info(
                    "Continuing execution loop at iteration %d", current_iteration
                )
            )

            # Build called tools prompt
            called_tools_prompt: UserMessage = UserMessage(parts=[TextPart(text="")])
            if called_tools:
                called_tools_prompt_parts: MutableSequence[TextPart | FilePart] = []
                for suggestion, result in called_tools.values():
                    if isinstance(result, FilePart):
                        called_tools_prompt_parts.extend(
                            [
                                TextPart(
                                    text=f"<info><tool_name>{suggestion.tool_name}</tool_name><args>{suggestion.args}</args><result>The following is a file that was generated by the tool:</info>"
                                ),
                                result,
                                TextPart(text="</result>"),
                            ]
                        )
                    else:
                        called_tools_prompt_parts.append(
                            TextPart(
                                text="""<info>
                                The following is a tool call made by the agent.
                                Only call it again if you think it's necessary.
                                </info>"""
                                + "\n"
                                + "\n".join(
                                    [
                                        f"""<tool_execution>
                                <tool_name>{suggestion.tool_name}</tool_name>
                                <args>{suggestion.args}</args>
                                <result>{result}</result>
                            </tool_execution>"""
                                    ]
                                )
                            )
                        )

                called_tools_prompt = UserMessage(
                    parts=[
                        TextPart(
                            text="".join(str(p.text) for p in called_tools_prompt_parts)
                        )
                    ]
                )

            _new_messages = (
                MessageSequence(context.message_history)
                .merge_with_last_user_message(called_tools_prompt)
                .elements
            )

            # Generate tool call response
            tool_call_generation = await generation_provider.generate_async(
                model=self.resolved_model,
                messages=_new_messages,
                generation_config=self.agent_config.generation_config,
                tools=all_tools,
            )

            context.update_token_usage(tool_call_generation.usage)

            # Check if agent called any tools
            if tool_call_generation.tool_calls_amount() == 0:
                _logger.bind_optional(
                    lambda log: log.info(
                        "No more tool calls, generating final response"
                    )
                )

                # Create final step
                final_step = Step(
                    step_type="generation",
                    iteration=current_iteration,
                    tool_execution_suggestions=[],
                    generation_text=tool_call_generation.text
                    or "Generating final response...",
                    token_usage=tool_call_generation.usage,
                )

                # Generate final response if needed
                if self.response_schema is not None or not tool_call_generation.text:
                    generation = await generation_provider.generate_async(
                        model=self.resolved_model,
                        messages=context.message_history,
                        response_schema=self.response_schema,
                        generation_config=self.agent_config.generation_config,
                    )

                    final_step.generation_text = generation.text
                    final_step.token_usage = generation.usage
                    context.update_token_usage(generation.usage)
                else:
                    generation = cast(Generation[T_Schema], tool_call_generation)

                final_step.mark_completed()
                context.add_step(final_step)
                context.complete_execution()

                return AgentRunOutput(
                    generation=generation,
                    context=context,
                    parsed=generation.parsed,
                )

            # Execute tools
            step = Step(
                step_type="tool_execution",
                iteration=current_iteration,
                tool_execution_suggestions=list(tool_call_generation.tool_calls),
                generation_text=tool_call_generation.text,
                token_usage=tool_call_generation.usage,
            )

            step_start_time = time.time()

            for tool_execution_suggestion in tool_call_generation.tool_calls:
                selected_tool = available_tools[tool_execution_suggestion.tool_name]

                tool_start_time = time.time()
                try:
                    tool_result = await selected_tool.call_async(
                        **tool_execution_suggestion.args
                    )
                    tool_execution_time = (time.time() - tool_start_time) * 1000

                    called_tools[tool_execution_suggestion.id] = (
                        tool_execution_suggestion,
                        tool_result,
                    )

                    step.add_tool_execution_result(
                        suggestion=tool_execution_suggestion,
                        result=tool_result,
                        execution_time_ms=tool_execution_time,
                        success=True,
                    )

                except ToolSuspensionError as suspension_error:
                    # Tool suspended - save state and return
                    await self._save_suspension_state(
                        context=context,
                        suspension_type="tool_execution",
                        tool_suggestion=tool_execution_suggestion,
                        current_iteration=current_iteration,
                        all_tools=all_tools,
                        called_tools=called_tools,
                        current_step=step.model_dump()
                        if hasattr(step, "model_dump")
                        else None,
                    )

                    suspension_mgr = (
                        self.suspension_manager or get_default_suspension_manager()
                    )
                    resumption_token = await suspension_mgr.suspend_execution(
                        context=context,
                        reason=suspension_error.reason,
                        approval_data=suspension_error.approval_data,
                        timeout_hours=suspension_error.timeout_seconds // 3600
                        if suspension_error.timeout_seconds
                        else 24,
                    )

                    return AgentRunOutput(
                        generation=None,
                        context=context,
                        parsed=cast(T_Schema, None),
                        is_suspended=True,
                        suspension_reason=suspension_error.reason,
                        resumption_token=resumption_token,
                    )

            # Complete step and continue
            step_duration = (time.time() - step_start_time) * 1000
            step.mark_completed(duration_ms=step_duration)
            context.add_step(step)

            current_iteration += 1

        # Max iterations reached
        error_message = f"Max tool calls exceeded after {self.agent_config.maxIterations} iterations"
        context.fail_execution(error_message)
        raise MaxToolCallsExceededError(error_message)

    async def _save_suspension_state(
        self,
        context: Context,
        suspension_type: str,
        tool_suggestion: ToolExecutionSuggestion | None = None,
        current_iteration: int | None = None,
        all_tools: Sequence[Tool[Any]] | None = None,
        called_tools: dict[str, tuple[ToolExecutionSuggestion, Any]] | None = None,
        current_step: dict[str, Any] | None = None,
    ) -> None:
        """
        Save the current execution state for proper resumption.

        This method saves all necessary state information so that execution
        can be resumed from the exact point where it was suspended.
        """
        suspension_state: dict[str, Any] = {
            "type": suspension_type,
            "timestamp": datetime.datetime.now().isoformat(),
        }

        if suspension_type == "tool_execution" and tool_suggestion:
            suspension_state["tool_suggestion"] = {
                "id": tool_suggestion.id,
                "tool_name": tool_suggestion.tool_name,
                "args": tool_suggestion.args,
            }
            suspension_state["current_iteration"] = current_iteration
            suspension_state["called_tools"] = {
                k: {
                    "suggestion": {
                        "id": v[0].id,
                        "tool_name": v[0].tool_name,
                        "args": v[0].args,
                    },
                    "result": str(v[1]),  # Serialize result as string
                }
                for k, v in (called_tools or {}).items()
            }
            suspension_state["current_step"] = current_step

        context.set_checkpoint_data("suspension_state", suspension_state)

    def _build_agent_run_output(
        self,
        *,
        context: Context,
        generation: Generation[T_Schema],
        performance_metrics: PerformanceMetrics | None = None,
    ) -> AgentRunOutput[T_Schema]:
        """
        Builds an AgentRunOutput object from the generation results.

        This internal method creates the standardized output structure of the agent,
        including artifacts, usage statistics, final context, and performance metrics.

        Args:
            context: The final context of the execution.
            generation: The Generation object produced by the provider.
            performance_metrics: Optional performance metrics collected during execution.

        Returns:
            AgentRunOutput[T_Schema]: The structured result of the agent execution.
        """
        parsed = generation.parsed

        return AgentRunOutput(
            generation=generation,
            context=context,
            parsed=parsed,
            performance_metrics=performance_metrics,
        )

    @classmethod
    def instructions2str(
        cls, instructions: str | Prompt | Callable[[], str] | Sequence[str]
    ) -> str:
        """
        Converts the instructions to a string.

        This internal method handles the different formats that instructions
        can have: simple string, callable that returns string, or sequence of strings.

        Args:
            instructions: The instructions in any supported format.

        Returns:
            str: The instructions converted to string.
        """
        if isinstance(instructions, str):
            return instructions
        elif isinstance(instructions, Prompt):
            return instructions.text
        elif callable(instructions):
            return instructions()
        else:
            return "".join(instructions)

    @classmethod
    def input2context(
        cls,
        input: AgentInput | Any,
        instructions: str,
    ) -> Context:
        """
        Converts user input to a Context object.

        This internal method converts the various supported input types to
        a standardized Context object that contains the messages to be processed.

        Supports a wide variety of input types, from simple strings to
        complex objects like DataFrames, images, files, and Pydantic models.

        Args:
            input: The input in any supported format.
            instructions: The agent instructions as a string.

        Returns:
            Context: A Context object containing the messages to be processed.
        """
        developer_message = DeveloperMessage(parts=[TextPart(text=instructions)])

        if isinstance(input, Context):
            # If it's already a Context, return it as is.
            input.add_developer_message(instructions)
            return input
        elif isinstance(input, UserMessage):
            # If it's a UserMessage, prepend the developer instructions.
            return Context(message_history=[developer_message, input])
        elif isinstance(input, str):
            # Handle plain string input
            return Context(
                message_history=[
                    developer_message,
                    UserMessage(parts=[TextPart(text=input)]),
                ]
            )
        elif isinstance(input, (TextPart, FilePart, Tool)):
            # Handle single message parts
            return Context(
                message_history=cast(
                    MutableSequence[DeveloperMessage | UserMessage | AssistantMessage],
                    [
                        developer_message,
                        UserMessage(
                            parts=cast(
                                MutableSequence[
                                    TextPart
                                    | FilePart
                                    | Tool[Any]
                                    | ToolExecutionSuggestion
                                    | ToolExecutionResult
                                ],
                                [input],
                            )
                        ),
                    ],
                )
            )

        # Sequence handling: Check for Message sequences or Part sequences
        # Explicitly check for Sequence for MyPy's benefit
        elif isinstance(input, Sequence) and not isinstance(input, (str, bytes)):  # pyright: ignore[reportUnnecessaryIsInstance]
            # Check if it's a sequence of Messages or Parts (AFTER specific types)
            if input and isinstance(
                input[0], (AssistantMessage, DeveloperMessage, UserMessage)
            ):
                # Sequence of Messages
                # Ensure it's a list of Messages for type consistency and prepend developer message
                message_history = [developer_message] + list(
                    cast(
                        Sequence[AssistantMessage | DeveloperMessage | UserMessage],
                        input,
                    )
                )
                return Context(message_history=message_history)
            elif input and isinstance(input[0], (TextPart, FilePart, Tool)):
                # Sequence of Parts
                # Ensure it's a list of the correct Part types
                valid_parts = cast(Sequence[TextPart | FilePart | Tool], input)
                return Context(
                    message_history=[
                        developer_message,
                        UserMessage(parts=list(valid_parts)),
                    ]
                )

        elif callable(input) and not isinstance(input, Tool):
            # Handle callable input (that's not a Tool)
            return Context(
                message_history=[
                    developer_message,
                    UserMessage(parts=[TextPart(text=str(input()))]),
                ]
            )
        # Handle pandas DataFrame if available
        elif HAS_PANDAS:
            try:
                import pandas as pd

                if isinstance(input, pd.DataFrame):
                    # Convert DataFrame to Markdown
                    return Context(
                        message_history=[
                            developer_message,
                            UserMessage(
                                parts=[TextPart(text=input.to_markdown() or "")]
                            ),
                        ]
                    )
            except ImportError:
                pass
        # Handle numpy arrays if available
        elif HAS_NUMPY:
            try:
                import numpy as np

                if isinstance(input, np.ndarray):
                    # Convert NumPy array to string representation
                    return Context(
                        message_history=[
                            developer_message,
                            UserMessage(
                                parts=[
                                    TextPart(
                                        text=np.array2string(
                                            cast(np.ndarray[Any, Any], input)
                                        )
                                    )
                                ]
                            ),
                        ]
                    )
            except ImportError:
                pass
        # Handle PIL images if available
        elif HAS_PIL:
            try:
                from PIL import Image

                if isinstance(input, Image.Image):
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
                    mime_type = mime_type_map.get(
                        img_format, f"image/{img_format.lower()}"
                    )

                    return Context(
                        message_history=[
                            developer_message,
                            UserMessage(
                                parts=[
                                    FilePart(
                                        data=img_byte_arr.getvalue(),
                                        mime_type=mime_type,
                                    )
                                ]
                            ),
                        ]
                    )
            except ImportError:
                pass
        elif isinstance(input, bytes):
            # Try decoding bytes, otherwise provide a description
            try:
                text = input.decode("utf-8")
            except UnicodeDecodeError:
                text = f"Input is binary data of size {len(input)} bytes."
            return Context(
                message_history=[
                    developer_message,
                    UserMessage(parts=[TextPart(text=text)]),
                ]
            )
        elif isinstance(input, (datetime.datetime, datetime.date, datetime.time)):
            # Convert datetime objects to ISO format string
            return Context(
                message_history=[
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
                        message_history=[
                            developer_message,
                            UserMessage(parts=[TextPart(text=file_content)]),
                        ]
                    )
                except Exception as e:
                    # Fallback to string representation if reading fails
                    return Context(
                        message_history=[
                            developer_message,
                            UserMessage(
                                parts=[
                                    TextPart(
                                        text=f"Failed to read file {input}: {str(e)}"
                                    )
                                ]
                            ),
                        ]
                    )
            else:
                # If it's not a file or doesn't exist, use the string representation
                return Context(
                    message_history=[
                        developer_message,
                        UserMessage(parts=[TextPart(text=str(input))]),
                    ]
                )
        elif isinstance(input, Prompt):
            return Context(
                message_history=[
                    developer_message,
                    UserMessage(parts=[TextPart(text=input.text)]),
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
                message_history=[
                    developer_message,
                    UserMessage(parts=[TextPart(text=text)]),
                ]
            )
        elif isinstance(input, ParsedFile):
            return Context(
                message_history=[
                    developer_message,
                    UserMessage(parts=[TextPart(text=input.md)]),
                ]
            )

        # Handle Pydantic models if available
        elif HAS_PYDANTIC:
            try:
                from pydantic import BaseModel as PydanticBaseModel

                if isinstance(input, PydanticBaseModel):
                    # Convert Pydantic model to JSON string
                    text = input.model_dump_json(indent=2)
                    return Context(
                        message_history=[
                            developer_message,
                            UserMessage(parts=[TextPart(text=f"```json\n{text}\n```")]),
                        ]
                    )
            except (ImportError, AttributeError):
                pass

        elif isinstance(input, (dict, list, tuple, set, frozenset)):
            # Convert dict, list, tuple, set, frozenset to JSON string
            try:
                # Use json.dumps for serialization
                text = json.dumps(
                    input, indent=2, default=str
                )  # Add default=str for non-serializable
            except TypeError:
                # Fallback to string representation if json fails
                text = f"Input is a collection: {str(cast(object, input))}"
            return Context(
                message_history=[
                    developer_message,
                    UserMessage(parts=[TextPart(text=f"```json\n{text}\n```")]),
                ]
            )

        # Fallback for any unhandled type
        # Convert to string representation as a last resort
        try:
            # Use safer type handling
            input_type_name = type(input).__name__  # type: ignore[reportGeneralTypeIssues, reportUnknownArgumentType]
            text = str(input)  # type: ignore[reportGeneralTypeIssues, reportUnknownArgumentType]
        except Exception:
            # Use safer type handling
            input_type_name = (
                "unknown"  # Fall back to a string if we can't get the type name
            )
            text = f"Input of type {input_type_name} could not be converted to string"

        return Context(  # type: ignore[reportGeneralTypeIssues, reportUnknownArgumentType]
            message_history=[
                developer_message,
                UserMessage(parts=[TextPart(text=text)]),  # type: ignore[reportGeneralTypeIssues, reportUnknownArgumentType]
            ]
        )

    def _get_tool_call_pattern_key(
        self, tool_name: str, args: Mapping[str, object]
    ) -> str:
        """Generate a deterministic key for tracking tool call patterns."""
        # Ensure consistent types for serialization
        normalized_args = {}
        for k, v in args.items():
            if isinstance(v, (int, float, bool, str, type(None))):
                normalized_args[k] = v
            else:
                normalized_args[k] = str(v)

        return f"{tool_name}:{json.dumps(normalized_args, sort_keys=True, default=str)}"

    def _format_tool_call_summary(
        self,
        called_tools: dict[str, tuple[ToolExecutionSuggestion, Any]],
        max_calls_to_show: int = 10,
    ) -> str:
        """Format a summary of tool calls made during execution."""
        if not called_tools:
            return "No tool calls were made."

        summary_lines = [f"Total tool calls made: {len(called_tools)}"]

        if len(called_tools) <= max_calls_to_show:
            summary_lines.append("\nTool calls made:")
            for i, (suggestion, result) in enumerate(called_tools.values(), 1):
                result_str = str(result)
                if len(result_str) > 200:
                    result_str = result_str[:197] + "..."

                summary_lines.append(
                    f"  {i}. {suggestion.tool_name}({json.dumps(suggestion.args, default=str)})"
                )
                summary_lines.append(f"     Result: {result_str}")
        else:
            show_count = max_calls_to_show // 2
            summary_lines.append(f"\nFirst {show_count} tool calls:")

            tool_items = list(called_tools.values())
            for i, (suggestion, result) in enumerate(tool_items[:show_count], 1):
                result_str = str(result)
                if len(result_str) > 100:
                    result_str = result_str[:97] + "..."
                summary_lines.append(
                    f"  {i}. {suggestion.tool_name}({json.dumps(suggestion.args, default=str)})"
                )
                summary_lines.append(f"     Result: {result_str}")

            summary_lines.append(
                f"\n... ({len(called_tools) - max_calls_to_show} tool calls omitted) ..."
            )

            summary_lines.append(f"\nLast {show_count} tool calls:")
            for i, (suggestion, result) in enumerate(
                tool_items[-show_count:], len(called_tools) - show_count + 1
            ):
                result_str = str(result)
                if len(result_str) > 100:
                    result_str = result_str[:97] + "..."
                summary_lines.append(
                    f"  {i}. {suggestion.tool_name}({json.dumps(suggestion.args, default=str)})"
                )
                summary_lines.append(f"     Result: {result_str}")

        return "\n".join(summary_lines)

    def _format_steps_summary(self, steps: Sequence[Step]) -> str:
        """Format a summary of execution steps."""
        if not steps:
            return "No execution steps recorded."

        summary_lines = [f"Total execution steps: {len(steps)}"]
        summary_lines.append("\nStep summary:")

        for i, step in enumerate(steps, 1):
            step_info = f"  {i}. Iteration {step.iteration}: {step.step_type}"

            if step.tool_execution_suggestions:
                tool_names = [
                    suggestion.tool_name
                    for suggestion in step.tool_execution_suggestions
                ]
                step_info += f" (tools: {', '.join(tool_names)})"

            if step.duration_ms:
                step_info += f" - {step.duration_ms:.1f}ms"

            if step.error_message:
                step_info += f" - ERROR: {step.error_message}"

            summary_lines.append(step_info)

        return "\n".join(summary_lines)

    def _analyze_tool_call_patterns(
        self,
        called_tools: dict[str, tuple[ToolExecutionSuggestion, Any]],
    ) -> str:
        """Analyze patterns in tool calls to identify potential issues."""
        if not called_tools:
            return "No tool calls to analyze."

        tool_usage: dict[str, Any] = {}
        repeated_calls: dict[str, Any] = {}

        for suggestion, _ in called_tools.values():
            tool_name = suggestion.tool_name
            tool_usage[tool_name] = tool_usage.get(tool_name, 0) + 1

            call_signature = f"{tool_name}({json.dumps(suggestion.args, sort_keys=True, default=str)})"
            repeated_calls[call_signature] = repeated_calls.get(call_signature, 0) + 1

        analysis_lines = ["TOOL CALL ANALYSIS:"]

        sorted_tools = sorted(tool_usage.items(), key=lambda x: x[1], reverse=True)
        analysis_lines.append(f"Most used tools: {dict(sorted_tools[:5])}")

        repeated = {call: count for call, count in repeated_calls.items() if count > 1}
        if repeated:
            analysis_lines.append(
                f"Repeated calls detected: {len(repeated)} unique calls repeated"
            )
            for call, count in sorted(
                repeated.items(), key=lambda x: x[1], reverse=True
            )[:3]:
                analysis_lines.append(f"  - {call}: {count} times")

        return "\n".join(analysis_lines)

    def __call__(
        self, input: AgentInput | Any
    ) -> AgentRunOutput[T_Schema] | AsyncIterator[AgentRunOutput[T_Schema]]:
        return self.run(input)

    def __add__(self, other: Agent[Any]) -> AgentTeam:
        from agentle.agents.agent_team import AgentTeam

        return AgentTeam(
            agents=[self, other],
            orchestrator_provider=self.generation_provider,
            orchestrator_model=self.resolved_model
            or self.generation_provider.default_model,
        )
