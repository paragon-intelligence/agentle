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
    model="gemini-2.0-flash",
    instructions="You are a weather agent that can answer questions about the weather.",
    tools=[get_weather],
)

output = weather_agent.run("Hello. What is the weather in Tokyo?")
```
"""

from __future__ import annotations

import datetime
import importlib.util
import json
import logging
import uuid
from collections.abc import (
    AsyncGenerator,
    Callable,
    Generator,
    MutableMapping,
    MutableSequence,
    Sequence,
)
from contextlib import asynccontextmanager, contextmanager
from typing import TYPE_CHECKING, Any, cast

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
from agentle.agents.agent_input import AgentInput
from agentle.agents.agent_run_output import AgentRunOutput
from agentle.agents.context import Context
from agentle.agents.errors.max_tool_calls_exceeded_error import (
    MaxToolCallsExceededError,
)
from agentle.agents.knowledge.static_knowledge import NO_CACHE, StaticKnowledge
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
from agentle.generations.models.messages.message import Message
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.providers.base.generation_provider import (
    GenerationProvider,
)
from agentle.generations.tools.tool import Tool

# from agentle.generations.tracing.langfuse import LangfuseObservabilityClient
from agentle.mcp.servers.mcp_server_protocol import MCPServerProtocol
from agentle.parsing.document_parser import DocumentParser
from agentle.parsing.factories.file_parser_default_factory import (
    file_parser_default_factory,
)
from agentle.parsing.parsed_document import ParsedDocument
from agentle.prompts.models.prompt import Prompt
from pathlib import Path
from io import BytesIO, StringIO

if TYPE_CHECKING:
    from agentle.agents.agent_team import AgentTeam

    from mcp.types import Tool as MCPTool


type WithoutStructuredOutput = None
type _ToolName = str

logger = logging.getLogger(__name__)


# Check for optional dependencies
def is_module_available(module_name: str) -> bool:
    """Check if a module is available without importing it."""
    return importlib.util.find_spec(module_name) is not None


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
            model="gemini-2.0-flash",
            instructions="You are a weather agent that can answer questions about the weather.",
            tools=[get_weather],
        )

        # Run the agent
        output = weather_agent.run("What is the weather in London?")
        ```
    """

    uid: uuid.UUID = Field(default_factory=uuid.uuid4)
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
    document_parser: DocumentParser | None = Field(default=None)
    """
    A document parser to be used by the agent. This will be used to parse the static
    knowledge documents, if provided.
    """

    generation_provider: GenerationProvider
    """
    The service provider of the agent
    """

    file_visual_description_provider: GenerationProvider | None = Field(default=None)
    """
    The service provider of the agent for visual description.
    """

    file_audio_description_provider: GenerationProvider | None = Field(default=None)
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
    model: str | None = Field(default=None)
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

    debug: bool = Field(default=False)
    """
    Whether to debug each agent step using the logger.
    """

    # Internal fields
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    @classmethod
    def from_agent_card(cls, agent_card: dict[str, Any]) -> "Agent[Any]":
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
                generation_provider = type("DummyProvider", (GenerationProvider,), {})

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
    def with_mcp_servers(self) -> Generator[None, None, None]:
        """
        Context manager to connect and clean up MCP servers.

        This context manager ensures that all MCP servers are connected before the
        code block is executed and cleaned up after completion, even in case of exceptions.

        Yields:
            None: Does not return a value, just manages the context.

        Example:
            ```python
            async with agent.with_mcp_servers():
                # Operations that require connection to MCP servers
                result = await agent.run_async("Query to server")
            # Servers are automatically disconnected here
            ```
        """
        for server in self.mcp_servers:
            run_sync(server.connect)
        try:
            yield
        finally:
            for server in self.mcp_servers:
                run_sync(server.cleanup)

    @asynccontextmanager
    async def with_mcp_servers_async(self) -> AsyncGenerator[None, None]:
        """
        Asynchronous context manager to connect and clean up MCP servers.

        This method ensures that all MCP servers are connected before the
        code block is executed and cleaned up after completion, even in case of exceptions.

        Yields:
            None: Does not return a value, just manages the context.

        Example:
            ```python
            async with agent.with_mcp_servers():
                # Operations that require connection to MCP servers
                result = await agent.run_async("Query to server")
            # Servers are automatically disconnected here
            ```
        """
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
        trace_params: TraceParams | None = None,
    ) -> AgentRunOutput[T_Schema]:
        """
        Runs the agent synchronously with the provided input.

        This method is a synchronous wrapper for run_async, allowing
        easy use in synchronous contexts.

        Args:
            input: The input for the agent, which can be of various types.
            timeout: Optional time limit in seconds for execution.
            trace_params: Optional trace parameters for observability purposes.

        Returns:
            AgentRunOutput[T_Schema]: The result of the agent execution.

        Example:
            ```python
            # Input as string
            result = agent.run("What is the weather in London?")

            # Input as UserMessage object
            from agentle.generations.models.messages.user_message import UserMessage
            from agentle.generations.models.message_parts.text import TextPart

            message = UserMessage(parts=[TextPart(text="What is the weather in London?")])
            result = agent.run(message)
            ```
        """
        return run_sync(
            self.run_async, timeout=timeout, input=input, trace_params=trace_params
        )

    async def run_async(
        self, input: AgentInput, trace_params: TraceParams | None = None
    ) -> AgentRunOutput[T_Schema]:
        """
        Runs the agent asynchronously with the provided input.

        This main method processes user input, interacts with the
        generation provider, and optionally calls tools until reaching a final response.

        The method supports both simple agents (without tools) and agents with
        tools that can perform iterative calls to solve complex tasks.

        Args:
            input: The input for the agent, which can be of various types.
            trace_params: Optional trace parameters for observability purposes.

        Returns:
            AgentRunOutput[T_Schema]: The result of the agent execution, possibly
                                     with a structured response according to the defined schema.

        Raises:
            MaxToolCallsExceededError: If the maximum number of tool calls is exceeded.

        Example:
            ```python
            # Asynchronous use
            result = await agent.run_async("What's the weather like in London?")

            # Processing the response
            response_text = result.artifacts[0].parts[0].text
            print(response_text)

            # With structured response schema
            if result.parsed:
                location = result.parsed.location
                weather = result.parsed.weather
            ```
        """
        _logger = Maybe(logger if self.debug else None)

        _logger.bind_optional(
            lambda log: log.info("Starting agent run with input type: %s", type(input))
        )
        generation_provider: GenerationProvider = self.generation_provider

        static_knowledge_prompt: str | None = None
        # Process static knowledge if any exists
        if self.static_knowledge:
            _logger.bind_optional(lambda log: log.debug("Processing static knowledge"))
            knowledge_contents: MutableSequence[str] = []
            for knowledge_item in self.static_knowledge:
                # Convert string to StaticKnowledge with NO_CACHE
                if isinstance(knowledge_item, str):
                    knowledge_item = StaticKnowledge(
                        content=knowledge_item, cache=NO_CACHE, parse_timeout=30
                    )

                # Process the knowledge item based on its content type
                content_to_parse = knowledge_item.content
                parsed_content = None
                cache_key = None
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
                        lambda log: log.debug("Using cache for knowledge item")
                    )
                    try:
                        # Import aiocache only when needed
                        from aiocache import cached

                        # Create a cache key based on the content
                        cache_key = f"static_knowledge_{hash(content_to_parse)}"

                        # Define a cached version of the parse function
                        @cached(  # type: ignore
                            ttl=None
                            if knowledge_item.cache == "infinite"
                            else knowledge_item.cache,
                            key=cache_key,
                        )
                        async def get_cached_parsed_content(
                            content_path: str,
                        ) -> ParsedDocument:
                            return await parser.parse_async(content_path)

                        # Get parsed content, either from cache or newly parsed
                        parsed_content = await get_cached_parsed_content(
                            content_to_parse
                        )
                    except ImportError:
                        # If aiocache is not installed, just parse normally
                        parsed_content = None

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

        instructions = self._convert_instructions_to_str(self.instructions)
        if static_knowledge_prompt:
            instructions += "\n\n" + static_knowledge_prompt

        context: Context = self._convert_input_to_context(
            input, instructions=instructions
        )
        _logger.bind_optional(
            lambda log: log.debug(
                "Converted input to context with %d messages", len(context.messages)
            )
        )

        _logger.bind_optional(
            lambda log: log.debug(
                "Using generation provider: %s", type(generation_provider).__name__
            )
        )

        mcp_tools: MutableSequence[MCPTool] = []
        if bool(self.mcp_servers):
            _logger.bind_optional(
                lambda log: log.debug("Getting tools from MCP servers")
            )
            for server in self.mcp_servers:
                tools = await server.list_tools()
                mcp_tools.extend(tools)
            _logger.bind_optional(
                lambda log: log.debug("Got %d tools from MCP servers", len(mcp_tools))
            )

        agent_has_tools = self.has_tools() or len(mcp_tools) > 0
        _logger.bind_optional(
            lambda log: log.debug("Agent has tools: %s", agent_has_tools)
        )
        if not agent_has_tools:
            _logger.bind_optional(
                lambda log: log.debug("No tools available, generating direct response")
            )
            generation: Generation[
                T_Schema
            ] = await generation_provider.create_generation_async(
                model=self.model,
                messages=context.messages,
                response_schema=self.response_schema,
                generation_config=self.config.generationConfig
                if trace_params is None
                else self.config.generationConfig.clone(new_trace_params=trace_params),
            )
            _logger.bind_optional(
                lambda log: log.debug(
                    "Generated response with %d tokens", generation.usage.total_tokens
                )
            )

            return AgentRunOutput(
                generation=generation,
                steps=context.steps,
                parsed=generation.parsed,
            )

        # Agent has tools. We must iterate until generate the final answer.

        all_tools: MutableSequence[Tool[Any]] = [
            Tool.from_mcp_tool(tool) for tool in mcp_tools
        ] + [
            Tool.from_callable(tool) if callable(tool) else tool for tool in self.tools
        ]
        _logger.bind_optional(
            lambda log: log.debug("Using %d tools in total", len(all_tools))
        )

        available_tools: MutableMapping[str, Tool[Any]] = {
            tool.name: tool for tool in all_tools
        }

        state = RunState[T_Schema].init_state()
        # Convert all tools in the array to Tool objects
        called_tools: dict[str, tuple[ToolExecutionSuggestion, Any]] = {}

        while state.iteration < self.config.maxIterations:
            _logger.bind_optional(
                lambda log: log.info(
                    "Starting iteration %d of %d",
                    state.iteration + 1,
                    self.config.maxIterations,
                )
            )
            # Remove the filtering of tools that have already been called
            # since we want to allow calling the same tool multiple times

            called_tools_prompt: str = (
                (
                    """<info>
                    The following are the other tool calls made by the agent:
                    </info>"""
                    + "\n"
                    + "\n".join(
                        [
                            f"""<tool_execution>
                    <tool_name>{suggestion.tool_name}</tool_name>
                    <args>{suggestion.args}</args>
                    <result>{result}</result>
                </tool_execution>"""
                            for suggestion, result in called_tools.values()
                        ]
                    )
                )
                if called_tools
                else ""
            )

            # We no longer decide if there are no more tools since we allow repeated tool calls
            # Instead, we'll rely on the agent to stop calling tools when it has all needed information
            _logger.bind_optional(
                lambda log: log.debug("Generating tool call response")
            )
            tool_call_generation = await generation_provider.create_generation_async(
                model=self.model,
                messages=MessageSequence(context.messages)
                .append_before_last_message(called_tools_prompt)
                .elements,
                generation_config=self.config.generationConfig,
                tools=all_tools,  # Use all tools in every iteration
            )
            _logger.bind_optional(
                lambda log: log.debug(
                    "Tool call generation completed with %d tool calls",
                    tool_call_generation.tool_calls_amount(),
                )
            )

            agent_didnt_call_any_tool = tool_call_generation.tool_calls_amount() == 0
            if agent_didnt_call_any_tool:
                _logger.bind_optional(
                    lambda log: log.info(
                        "Agent didn't call any tool, generating final response"
                    )
                )
                # Only make another call if we need structured output or didn't get text
                if self.response_schema is not None or not tool_call_generation.text:
                    _logger.bind_optional(
                        lambda log: log.debug("Generating structured response")
                    )
                    generation = await generation_provider.create_generation_async(
                        model=self.model,
                        messages=context.messages,
                        response_schema=self.response_schema,
                        generation_config=self.config.generationConfig,
                    )
                    _logger.bind_optional(
                        lambda log: log.debug("Final generation complete")
                    )
                    return self._build_agent_run_output(
                        context=context, generation=generation
                    )

                # If we got text and don't need structure, use what we have
                _logger.bind_optional(
                    lambda log: log.debug("Using existing text response")
                )
                return self._build_agent_run_output(
                    generation=cast(Generation[T_Schema], tool_call_generation),
                    context=context,
                )

            # Agent called one tool. We must call the tool and update the state.
            _logger.bind_optional(
                lambda log: log.info(
                    "Processing %d tool calls", len(tool_call_generation.tool_calls)
                )
            )

            for tool_execution_suggestion in tool_call_generation.tool_calls:
                _logger.bind_optional(
                    lambda log: log.debug(
                        "Executing tool: %s with args: %s",
                        tool_execution_suggestion.tool_name,
                        tool_execution_suggestion.args,
                    )
                )
                tool_result = available_tools[tool_execution_suggestion.tool_name].call(
                    **tool_execution_suggestion.args
                )
                _logger.bind_optional(
                    lambda log: log.debug("Tool execution result: %s", tool_result)
                )
                called_tools[tool_execution_suggestion.id] = (  # here
                    tool_execution_suggestion,
                    tool_result,
                )

            state.update(
                last_response=tool_call_generation.text,
                tool_calls_amount=tool_call_generation.tool_calls_amount(),
                iteration=state.iteration + 1,
                token_usage=tool_call_generation.usage,
            )

        _logger.bind_optional(
            lambda log: log.error(
                "Max tool calls exceeded after %d iterations", self.config.maxIterations
            )
        )
        raise MaxToolCallsExceededError(
            f"Max tool calls exceeded after {self.config.maxIterations} iterations"
        )

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
        )

    def _build_agent_run_output(
        self,
        *,
        context: Context,
        generation: Generation[T_Schema],
    ) -> AgentRunOutput[T_Schema]:
        """
        Builds an AgentRunOutput object from the generation results.

        This internal method creates the standardized output structure of the agent,
        including artifacts, usage statistics, and final context.

        Args:
            artifacts: Optional sequence of pre-built artifacts.
            artifact_name: Name of the artifact to be created (if artifacts is not provided).
            artifact_description: Description of the artifact to be created.
            artifact_metadata: Optional metadata for the artifact.
            context: The final context of the execution.
            generation: The Generation object produced by the provider.
            task_status: The state of the task (default: COMPLETED).
            append: Whether the artifact should be appended to existing artifacts.
            last_chunk: Whether this is the last chunk of the artifact.

        Returns:
            AgentRunOutput[T_Schema]: The structured result of the agent execution.
        """
        parsed = generation.parsed

        return AgentRunOutput(
            generation=generation,
            steps=context.steps,
            parsed=parsed,
        )

    def _convert_instructions_to_str(
        self, instructions: str | Callable[[], str] | Sequence[str]
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
        elif callable(instructions):
            return instructions()
        else:
            return "".join(instructions)

    def _convert_input_to_context(
        self,
        input: AgentInput,
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
        # Handle pandas DataFrame if available
        elif HAS_PANDAS:
            try:
                import pandas as pd

                if isinstance(input, pd.DataFrame):
                    # Convert DataFrame to Markdown
                    return Context(
                        messages=[
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
                        messages=[
                            developer_message,
                            UserMessage(parts=[TextPart(text=np.array2string(input))]),
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
                        messages=[
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
                messages=[
                    developer_message,
                    UserMessage(parts=[TextPart(text=text)]),
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
                                        text=f"Failed to read file {input}: {str(e)}"
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
        elif isinstance(input, Prompt):
            return Context(
                messages=[
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
                messages=[
                    developer_message,
                    UserMessage(parts=[TextPart(text=text)]),
                ]
            )
        elif isinstance(input, ParsedDocument):
            return Context(
                messages=[
                    developer_message,
                    UserMessage(parts=[TextPart(text=input.md)]),
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

        # Handle Pydantic models if available
        elif HAS_PYDANTIC:
            try:
                from pydantic import BaseModel as PydanticBaseModel

                if isinstance(input, PydanticBaseModel):
                    # Convert Pydantic model to JSON string
                    text = input.model_dump_json(indent=2)
                    return Context(
                        messages=[
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
                text = f"Input is a collection: {str(input)}"
            return Context(
                messages=[
                    developer_message,
                    UserMessage(parts=[TextPart(text=f"```json\n{text}\n```")]),
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

    def __add__(self, other: Agent[Any]) -> AgentTeam:
        from agentle.agents.agent_team import AgentTeam

        return AgentTeam(
            agents=[self, other],
            orchestrator_provider=self.generation_provider,
            orchestrator_model=self.model or self.generation_provider.default_model,
        )
