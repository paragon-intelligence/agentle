"""
Chain of Thought reasoning implementation for the Agentle framework.

This module provides the primary model for representing and working with
chain-of-thought reasoning processes in AI generations. The Chain of Thought
technique makes the reasoning process explicit by breaking down complex problem-solving
into discrete, observable steps, leading to a final answer.

This structured approach offers several benefits:
- Improved reasoning transparency for complex tasks
- Better error detection and debugging of model reasoning
- Support for step-by-step verification of logical processes
- Enhanced explainability for regulatory or user trust requirements

The module implements a Pydantic model for structured Chain of Thought representation
with multilingual output formatting capabilities.
"""

from __future__ import annotations

import copy
import datetime
import logging
import uuid
from collections.abc import Callable, Sequence
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    NotRequired,
    TypedDict,
    overload,
)

from rsb.models.base_model import BaseModel
from rsb.models.config_dict import ConfigDict
from rsb.models.field import Field
from rsb.models.private_attr import PrivateAttr

if TYPE_CHECKING:
    from mcp.types import Tool as MCPTool


logger = logging.getLogger(__name__)


class Tool[T_Output = Any](BaseModel):
    """
    A callable tool that can be used by AI models to perform specific functions.

    The Tool class represents a callable function with associated metadata such as name,
    description, and parameter specifications. Tools are the primary mechanism for enabling
    AI agents to interact with external systems, retrieve information, and perform actions.

    A Tool instance can be created either directly from a Python callable function using the
    `from_callable` class method, or from an MCP (Model Control Protocol) tool format using
    the `from_mcp_tool` class method.

    The class is generic with a T_Output type parameter that represents the return type of
    the underlying callable function.

    Attributes:
        type: Literal field that identifies this as a tool, always set to "tool".
        name: Human-readable name of the tool.
        description: Human-readable description of what the tool does.
        parameters: Dictionary of parameter specifications for the tool.
        _callable_ref: Private attribute storing the callable function.
        needs_human_confirmation: Flag indicating if human confirmation is needed before execution.

    Examples:
        ```python
        # Create a tool directly with parameters
        calculator_tool = Tool(
            name="calculate",
            description="Performs arithmetic calculations",
            parameters={
                "expression": {
                    "type": "string",
                    "description": "The arithmetic expression to evaluate",
                    "required": True
                }
            }
        )

        # Create a tool from a function
        def fetch_user_data(user_id: str) -> dict:
            \"\"\"Retrieve user data from the database\"\"\"
            # Implementation would connect to a database
            return {"id": user_id, "name": "Example User"}

        user_data_tool = Tool.from_callable(fetch_user_data)
        ```
    """

    type: Literal["tool"] = Field(
        default="tool",
        description="Discriminator field identifying this as a tool object.",
        examples=["tool"],
    )

    name: str = Field(
        description="Human-readable name of the tool, used for identification and display.",
        examples=["get_weather", "search_database", "calculate_expression"],
    )

    description: str | None = Field(
        default=None,
        description="Human-readable description of what the tool does and how to use it.",
        examples=[
            "Get the current weather for a specified location",
            "Search the database for records matching the query",
        ],
    )

    parameters: dict[str, object] = Field(
        description="Dictionary of parameter specifications for the tool, including types, descriptions, and constraints.",
        examples=[
            {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                    "required": True,
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "default": "celsius",
                },
            }
        ],
    )

    _callable_ref: Callable[..., T_Output] | None = PrivateAttr(default=None)

    needs_human_confirmation: bool = Field(
        default=False,
        description="Flag indicating whether human confirmation is required before executing this tool.",
        examples=[True, False],
    )

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    @property
    def text(self) -> str:
        """
        Generates a human-readable text representation of the tool.

        Returns:
            str: A formatted string containing the tool name, description, and parameters.

        Example:
            ```python
            weather_tool = Tool(
                name="get_weather",
                description="Get weather for a location",
                parameters={"location": {"type": "string", "required": True}}
            )

            print(weather_tool.text)
            # Output:
            # Tool: get_weather
            # Description: Get weather for a location
            # Parameters: {'location': {'type': 'string', 'required': True}}
            ```
        """
        return f"Tool: {self.name}\nDescription: {self.description}\nParameters: {self.parameters}"

    def call(self, **kwargs: object) -> T_Output:
        """
        Executes the underlying function with the provided arguments.

        This method calls the function referenced by the `_callable_ref` attribute
        with the provided keyword arguments. It raises a ValueError if the Tool
        was not created with a callable reference.

        Args:
            **kwargs: Keyword arguments to pass to the underlying function.

        Returns:
            T_Output: The result of calling the underlying function.

        Raises:
            ValueError: If the Tool does not have a callable reference.

        Example:
            ```python
            def add(a: int, b: int) -> int:
                \"\"\"Add two numbers\"\"\"
                return a + b

            add_tool = Tool.from_callable(add)
            result = add_tool.call(a=5, b=3)
            print(result)  # Output: 8
            ```
        """
        if self._callable_ref is None:
            raise ValueError(
                'Tool is not callable because the "_callable_ref" instance variable is not set'
            )

        return self._callable_ref(**kwargs)

    @classmethod
    def from_mcp_tool(cls, mcp_tool: MCPTool) -> Tool[T_Output]:
        """
        Creates a Tool instance from an MCP Tool.

        This class method constructs a Tool from the Model Control Protocol (MCP)
        Tool format, extracting the name, description, and parameter schema.

        Args:
            mcp_tool: An MCP Tool object with name, description, and inputSchema.

        Returns:
            Tool[T_Output]: A new Tool instance.

        Example:
            ```python
            from mcp.types import Tool as MCPTool

            # Assuming an MCP tool object is available
            mcp_tool = MCPTool(
                name="search",
                description="Search for information",
                inputSchema={"query": {"type": "string", "required": True}}
            )

            search_tool = Tool.from_mcp_tool(mcp_tool)
            ```
        """
        return cls(
            name=mcp_tool.name,
            description=mcp_tool.description,
            parameters=mcp_tool.inputSchema,
        )

    @classmethod
    def from_callable(
        cls,
        _callable: Callable[..., T_Output],
        /,
    ) -> Tool[T_Output]:
        """
        Creates a Tool instance from a callable function.

        This class method analyzes a function's signature, including its name,
        docstring, parameter types, and default values, to create a Tool instance.
        The resulting Tool encapsulates the function and its metadata.

        Args:
            _callable: A callable function to wrap as a Tool.

        Returns:
            Tool[T_Output]: A new Tool instance with the callable function set as its reference.

        Example:
            ```python
            def search_database(query: str, limit: int = 10) -> list[dict]:
                \"\"\"Search the database for records matching the query\"\"\"
                # Implementation would typically search a database
                return [{"id": 1, "result": f"Result for {query}"}] * min(limit, 100)

            db_search_tool = Tool.from_callable(search_database)

            # The resulting tool will have:
            # - name: "search_database"
            # - description: "Search the database for records matching the query"
            # - parameters: {
            #     "query": {"type": "str", "required": True},
            #     "limit": {"type": "int", "default": 10}
            # }
            ```
        """
        import inspect

        name = getattr(_callable, "__name__", "anonymous_function")
        description = _callable.__doc__ or "No description available"

        # Extrair informações dos parâmetros da função
        parameters: dict[str, object] = {}
        signature = inspect.signature(_callable)

        for param_name, param in signature.parameters.items():
            # Ignorar parâmetros do tipo self/cls para métodos
            if (
                param_name in ("self", "cls")
                and param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
            ):
                continue

            param_info: dict[str, object] = {"type": "object"}

            # Adicionar informações de tipo se disponíveis
            if param.annotation != inspect.Parameter.empty:
                param_type = (
                    str(param.annotation).replace("<class '", "").replace("'>", "")
                )
                param_info["type"] = param_type

            # Adicionar valor padrão se disponível
            if param.default != inspect.Parameter.empty:
                param_info["default"] = param.default

            # Determinar se o parâmetro é obrigatório
            if param.default == inspect.Parameter.empty and param.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            ):
                param_info["required"] = True

            parameters[param_name] = param_info

        instance = cls(
            name=name,
            description=description,
            parameters=parameters,
        )

        # Definir o atributo privado após a criação da instância
        instance._callable_ref = _callable

        return instance


class FilePart(BaseModel):
    """
    Represents a file attachment part of a message.

    This class handles binary file data with appropriate MIME type validation.
    """

    data: bytes = Field(description="The binary content of the file.")

    mime_type: str = Field(
        description="The MIME type of the file, must be a valid MIME type from Python's mimetypes module."
    )

    type: Literal["file"] = Field(
        default="file",
        description="Discriminator field to identify this as a file message part.",
    )

    @property
    def text(self) -> str:
        """
        Returns a text representation of the file part.

        Returns:
            str: A text representation containing the MIME type.
        """
        return f"<file>\n{self.mime_type}\n </file>"

    def __post_init__(self) -> None:
        """
        Validates that the provided MIME type is official.

        Raises:
            ValueError: If the MIME type is not in the list of official MIME types.
        """
        import mimetypes

        allowed_mimes = mimetypes.types_map.values()
        mime_type_unknown = self.mime_type not in allowed_mimes
        if mime_type_unknown:
            raise ValueError(
                f"The provided MIME ({self.mime_type}) is not in the list of official mime types: {allowed_mimes}"
            )


class TextPart(BaseModel):
    """
    Represents a plain text part of a message.

    This class is used for textual content within messages in the system.
    """

    text: str = Field(description="The textual content of the message part.")

    type: Literal["text"] = Field(
        default="text",
        description="Discriminator field to identify this as a text message part.",
    )


class ToolExecutionSuggestion(BaseModel):
    """
    Represents a suggestion to execute a specific tool.

    This class is used to model tool execution suggestions within messages,
    including the tool name and arguments.
    """

    type: Literal["tool_execution"] = Field(
        default="tool_execution",
        description="Discriminator field to identify this as a tool execution suggestion.",
    )

    id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique identifier for this tool execution suggestion.",
    )

    tool_name: str = Field(description="The name of the tool to be executed.")

    args: dict[str, object] = Field(
        default_factory=dict,
        description="The arguments to pass to the tool during execution.",
    )

    model_config = ConfigDict(frozen=True)

    @property
    def text(self) -> str:
        """
        Returns a text representation of the tool execution suggestion.

        Returns:
            str: A text representation containing the tool name and arguments.
        """
        return f"Tool: {self.tool_name}\nArgs: {self.args}"


type Part = Annotated[
    TextPart | FilePart | Tool | ToolExecutionSuggestion, Field(discriminator="type")
]


class AssistantMessage(BaseModel):
    """
    Represents a message from an assistant in the system.

    This class can contain a sequence of different message parts including
    text, files, and tool execution suggestions.
    """

    role: Literal["assistant"] = Field(
        default="assistant",
        description="Discriminator field to identify this as an assistant message. Always set to 'assistant'.",
    )

    parts: Sequence[TextPart | FilePart | ToolExecutionSuggestion] = Field(
        description="The sequence of message parts that make up this assistant message."
    )


class DeveloperMessage(BaseModel):
    """
    Represents a message from a developer in the system.

    This class can contain a sequence of different message parts including
    text, files, and tools.
    """

    role: Literal["developer"] = Field(
        default="developer",
        description="Discriminator field to identify this as a developer message. Always set to 'developer'.",
    )

    parts: Sequence[TextPart | FilePart | Tool] = Field(
        description="The sequence of message parts that make up this developer message."
    )


class GeneratedAssistantMessage[T](BaseModel):
    """
    Represents a message generated by an assistant with parsed content.

    This class extends the concept of an assistant message with a parsed
    representation of the message content of type T. It supports generic typing
    to allow for different types of parsed content.
    """

    role: Literal["assistant"] = Field(
        default="assistant",
        description="Discriminator field to identify this as an assistant message. Always set to 'assistant'.",
    )

    parts: Sequence[TextPart | ToolExecutionSuggestion] = Field(
        description="The sequence of message parts that make up this generated assistant message."
    )

    parsed: T = Field(
        description="The parsed representation of the message content of type T."
    )

    @property
    def tool_calls(self) -> Sequence[ToolExecutionSuggestion]:
        """
        Returns all tool execution suggestions contained in this message.

        Returns:
            Sequence[ToolExecutionSuggestion]: A sequence of tool execution suggestions.
        """
        return [
            part for part in self.parts if isinstance(part, ToolExecutionSuggestion)
        ]

    @property
    def text(self) -> str:
        """
        Returns the concatenated text representation of all parts in this message.

        Returns:
            str: The concatenated text of all message parts.
        """
        return "".join(part.text for part in self.parts)


class UserMessage(BaseModel):
    """
    Represents a message from a user in the system.

    This class can contain a sequence of different message parts including
    text, files, tools, and tool execution suggestions.
    """

    role: Literal["user"] = Field(
        default="user",
        description="Discriminator field to identify this as a user message. Always set to 'user'.",
    )

    parts: Sequence[TextPart | FilePart | Tool[Any] | ToolExecutionSuggestion] = Field(
        description="The sequence of message parts that make up this user message."
    )


type Message = Annotated[
    AssistantMessage | DeveloperMessage | UserMessage, Field(discriminator="role")
]


class Choice[T](BaseModel):
    """
    A single candidate response from an AI generation.

    Choice objects represent individual responses when a model provides multiple
    alternative completions for the same input. Each Choice contains an index
    indicating its position in the array of choices and a message containing
    the actual content generated by the model.

    The generic type parameter T allows for typed access to parsed structured
    data when the model response has been parsed into a specific schema.

    Attributes:
        index: Zero-based position of this choice in the array of choices.
        message: The response content generated by the model, potentially
            containing both text and structured data of type T.
    """

    index: int = Field(
        description="The zero-based index position of this choice in the sequence of responses. Used for identification and ordering when multiple alternative choices are generated.",
        ge=0,  # Must be greater than or equal to 0
        examples=[0, 1, 2],
    )

    message: GeneratedAssistantMessage[T] = Field(
        description="The complete message content produced by the AI model for this choice. Contains all generated text, tool calls, and structured data (if parsed). This represents the actual output content for this alternative response.",
        kw_only=False,
    )


class GenerationConfig(BaseModel):
    """
    Configuration parameters for controlling AI generation behavior.

    This class defines the various parameters that can be adjusted to control
    how AI models generate text. It includes common parameters supported across
    different providers (like temperature and top_p), as well as settings for
    tracing, timeouts, and provider-specific options.

    Attributes:
        temperature: Controls randomness in generation. Higher values (e.g., 0.8) make output
            more random, lower values (e.g., 0.2) make it more deterministic. Range 0-1.
        max_output_tokens: Maximum number of tokens to generate in the response.
        n: Number of alternative completions to generate.
        top_p: Nucleus sampling parameter - considers only the top p% of probability mass.
            Range 0-1.
        top_k: Only sample from the top k tokens at each step.
        google_generation_kwargs: Additional parameters specific to Google AI models.
        trace_params: Parameters for tracing the generation for observability.
        timeout: Maximum time in seconds to wait for a generation before timing out.
    """

    temperature: float | None = Field(
        default=None,
        description="Controls randomness in text generation. Higher values (e.g., 0.8) produce more diverse and creative outputs, while lower values (e.g., 0.2) produce more focused and deterministic results. Setting to 0 means deterministic output.",
        ge=0.0,
        le=1.0,
        examples=[0.0, 0.5, 0.7, 1.0],
    )
    max_output_tokens: int | None = Field(
        default=None,
        description="Maximum number of tokens the model will generate in its response. Helps control response length and prevent excessively long outputs. Setting too low may truncate important information.",
        gt=0,
        examples=[256, 1024, 4096],
    )
    n: int = Field(
        default=1,
        description="Number of alternative completions to generate for the same prompt. Useful for providing different response options or for techniques like self-consistency that require multiple generations.",
        ge=1,
        examples=[1, 3, 5],
    )
    top_p: float | None = Field(
        default=None,
        description="Nucleus sampling parameter that controls diversity by considering tokens comprising the top_p probability mass. A value of 0.9 means only considering tokens in the top 90% of probability mass. Lower values increase focus, higher values increase diversity.",
        ge=0.0,
        le=1.0,
        examples=[0.9, 0.95, 1.0],
    )
    top_k: float | None = Field(
        default=None,
        description="Limits token selection to the top k most likely tokens at each generation step. Helps filter out low-probability tokens. Lower values restrict creativity but increase focus and coherence.",
        ge=0.0,
        examples=[10, 40, 100],
    )
    google_generation_kwargs: dict[str, object] | None = Field(
        default=None,
        description="Additional parameters specific to Google AI model generation. Allows passing provider-specific parameters that aren't standardized across all providers in the Agentle framework.",
    )
    trace_params: TraceParams = Field(
        default_factory=lambda: TraceParams(),
        description="Configuration for tracing and observability of the generation process. Controls what metadata is captured about the generation for monitoring, debugging, and analysis purposes.",
    )
    timeout: float | None = Field(
        default=None,
        description="Maximum time in seconds to wait for a generation response before timing out. Helps prevent indefinite waits for slow or stuck generations. Recommended to set based on expected model and prompt complexity.",
        gt=0,
        examples=[10.0, 30.0, 60.0],
    )

    def clone(
        self,
        *,
        new_temperature: float | None = None,
        new_max_output_tokens: int | None = None,
        new_n: int | None = None,
        new_top_p: float | None = None,
        new_top_k: float | None = None,
        new_google_generation_kwargs: dict[str, object] | None = None,
        new_trace_params: TraceParams | None = None,
        new_timeout: float | None = None,
    ) -> GenerationConfig:
        """
        Creates a new GenerationConfig with optionally updated parameters.

        This method allows creating a modified copy of the current configuration
        without altering the original object, following the immutable pattern.

        Args:
            new_temperature: New temperature value, if provided.
            new_max_output_tokens: New maximum output tokens value, if provided.
            new_n: New number of completions value, if provided.
            new_top_p: New top_p value, if provided.
            new_top_k: New top_k value, if provided.
            new_google_generation_kwargs: New Google-specific parameters, if provided.
            new_trace_params: New trace parameters, if provided.
            new_timeout: New timeout value, if provided.

        Returns:
            A new GenerationConfig instance with the specified updates applied.
        """
        return GenerationConfig(
            temperature=new_temperature
            if new_temperature is not None
            else self.temperature,
            max_output_tokens=new_max_output_tokens
            if new_max_output_tokens is not None
            else self.max_output_tokens,
            n=new_n if new_n is not None else self.n,
            top_p=new_top_p if new_top_p is not None else self.top_p,
            top_k=new_top_k if new_top_k is not None else self.top_k,
            google_generation_kwargs=new_google_generation_kwargs
            if new_google_generation_kwargs is not None
            else self.google_generation_kwargs,
            trace_params=new_trace_params
            if new_trace_params is not None
            else self.trace_params,
            timeout=new_timeout if new_timeout is not None else self.timeout,
        )

    class Config:
        arbitrary_types_allowed = True


class TraceParams(TypedDict, total=False):
    """Parameters for tracking and analyzing LLM interactions.

    Traces provide a way to capture and analyze AI model interactions for
    purposes such as monitoring, debugging, analytics, and compliance.
    These parameters control what information is captured in a trace and
    how it's identified and categorized.

    All fields are optional, allowing for flexible configuration based on
    specific tracing needs and requirements.

    Attributes:
        name: Unique identifier for the trace
        input: Input parameters for the traced operation
        output: Result of the traced operation
        user_id: ID of user initiating the request
        session_id: Grouping identifier for related traces
        version: Version of the trace. Can be used for tracking changes
        release: Deployment release identifier
        metadata: Custom JSON-serializable metadata
        tags: Categorization labels for filtering
        public: Visibility flag for trace data
        parent_trace_id: ID of parent trace for establishing trace hierarchy

    Example:
        >>> trace = TraceParams(
        ...     name="customer_support",
        ...     tags=["urgent", "billing"]
        ... )
    """

    name: NotRequired[str]
    input: NotRequired[Any]
    output: NotRequired[Any]
    user_id: NotRequired[str]
    session_id: NotRequired[str]
    version: NotRequired[str]
    release: NotRequired[str]
    metadata: NotRequired[Any]
    tags: NotRequired[Sequence[str]]
    public: NotRequired[bool]
    parent_trace_id: NotRequired[str]


class Usage(BaseModel):
    """
    Tracks and calculates token consumption in AI model interactions.

    This class encapsulates metrics about token usage in AI generation operations,
    tracking both tokens sent to the model (prompt) and tokens generated by the
    model (completion). These metrics are crucial for monitoring resource usage,
    estimating costs, and optimizing interactions with AI models.

    The class provides utility methods for calculating total token usage and
    supports aggregation of usage data across multiple operations through
    arithmetic operations.

    Attributes:
        prompt_tokens: Number of tokens in the prompt sent to the model.
        completion_tokens: Number of tokens generated by the model in its response.

    Methods:
        total_tokens: Calculates the total number of tokens used (prompt + completion).
        __add__: Enables adding two Usage objects together, aggregating their token counts.
        __radd__: Supports adding Usage objects in sum() operations, starting from 0.
        zero: Creates an empty Usage object with zero tokens.
    """

    prompt_tokens: int = Field(
        default=0,
        description="Number of tokens in the prompt sent to the AI model. Represents the input token count that affects context window usage and factors into pricing calculations.",
        ge=0,
        examples=[10, 500, 2000],
    )

    completion_tokens: int = Field(
        default=0,
        description="Number of tokens generated by the AI model in its response. Represents the output token count that factors into pricing calculations and affects generation time.",
        ge=0,
        examples=[5, 250, 1000],
    )

    @property
    def total_tokens(self) -> int:
        """
        Calculate the total number of tokens used in the operation.

        Returns:
            The sum of prompt_tokens and completion_tokens.
        """
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        """
        Add two Usage objects together, aggregating their token counts.

        Args:
            other: Another Usage object to add to this one.

        Returns:
            A new Usage object with summed token counts.
        """
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )

    def __radd__(self, other: int | Usage) -> Usage:
        """
        Support adding Usage objects in sum() operations, starting from 0.

        Args:
            other: Typically 0 when used in sum().

        Returns:
            This Usage object if other is 0, otherwise performs normal addition.
        """
        if other == 0:
            return self
        return self.__add__(other)  # type: ignore

    @classmethod
    def zero(cls) -> Usage:
        """
        Create an empty Usage object with zero tokens.

        Returns:
            A new Usage object with all token counts set to zero.
        """
        return cls(prompt_tokens=0, completion_tokens=0)


class Generation[T](BaseModel):
    """
    Primary container for AI-generated content with metadata.

    The Generation class encapsulates a complete response from an AI model,
    including the generated content, metadata about the generation process,
    usage statistics, and potential structured data of type T.

    This class serves as the central return type for all provider implementations,
    ensuring a consistent interface regardless of which AI provider is being used.
    It supports multiple response choices (alternatives), type-safe structured data
    access, and convenient accessors for commonly needed information.

    The generic type parameter T allows for structured output parsing, enabling
    type-safe access to parsed data when a response_schema is provided.

    Attributes:
        elapsed_time: Time taken to generate the response
        id: Unique identifier for this generation
        object: Type identifier, always "chat.generation"
        created: Timestamp when this generation was created
        model: Identifier of the model that produced this generation
        choices: Sequence of alternative responses from the model
        usage: Token usage statistics for this generation
    """

    id: uuid.UUID = Field(
        description="Unique identifier for tracking and referencing this specific generation throughout the system. Used for logging, debugging, and associating generations with specific requests.",
        examples=[uuid.uuid4(), uuid.uuid4(), uuid.uuid4()],
    )
    object: Literal["chat.generation"] = Field(
        description="Type discriminator that identifies this object as a generation. Always set to 'chat.generation' to support polymorphic handling of different response types.",
        examples=["chat.generation"],
    )
    created: datetime.datetime = Field(
        description="ISO 8601 timestamp when this generation was created. Useful for tracking generation history, calculating processing time, and implementing time-based features.",
        examples=[
            datetime.datetime.now(),
            datetime.datetime.now() - datetime.timedelta(minutes=5),
        ],
    )
    model: str = Field(
        description="Identifier string for the AI model that produced this generation. Includes provider and model name/version information to enable model-specific handling and analytics.",
        examples=["gpt-4-turbo", "claude-3-sonnet", "llama-3-70b-instruct"],
    )
    choices: Sequence[Choice[T]] = Field(
        description="Collection of alternative responses from the model when multiple completions are requested. Each choice contains a generated message with text content, tool calls, and optional parsed structured data.",
    )
    usage: Usage = Field(
        description="Token usage statistics for tracking resource consumption and cost. Contains counts for tokens in the prompt and completion, enabling precise usage tracking and cost estimation across providers.",
        examples=[
            Usage(prompt_tokens=150, completion_tokens=50),
            Usage(prompt_tokens=800, completion_tokens=200),
        ],
    )

    @property
    def parsed(self) -> T:
        """
        Get the parsed structured data from the first choice.

        This is a convenience property that returns the parsed data from the
        first choice in the choices sequence. It's useful when you only have
        one choice and want direct access to the parsed data.

        Returns:
            T: The parsed structured data from the first choice

        Raises:
            ValueError: If there are multiple choices, as it's ambiguous
                which one to use
        """
        if len(self.choices) > 1:
            raise ValueError(
                "Choices list is > 1. Coudn't determine the parsed "
                + "model to obtain. Please, use the get_parsed "
                + "method, instead, passing the choice number "
                + "you want to get the parsed model."
            )

        return self.get_parsed(0)

    @property
    def parts(self) -> Sequence[TextPart | ToolExecutionSuggestion]:
        """
        Get the message parts from the first choice.

        This is a convenience property that returns the message parts from the
        first choice in the choices sequence. It includes both text parts and
        tool execution suggestions.

        Returns:
            Sequence[TextPart | ToolExecutionSuggestion]: The message parts
                from the first choice
        """
        if len(self.choices) > 1:
            logger.warning(
                "WARNING: choices list is > 1. Coudn't determine the parts. Returning the first choice parts."
            )

        return self.get_message_parts(0)

    def get_message_parts(
        self, choice: int
    ) -> Sequence[TextPart | ToolExecutionSuggestion]:
        """
        Get the message parts from a specific choice.

        Args:
            choice: The index of the choice to get message parts from

        Returns:
            Sequence[TextPart | ToolExecutionSuggestion]: The message parts
                from the specified choice
        """
        return self.choices[choice].message.parts

    @property
    def tool_calls(self) -> Sequence[ToolExecutionSuggestion]:
        """
        Get tool execution suggestions from the first choice.

        This is a convenience property that returns only the tool execution
        suggestions from the first choice in the choices sequence.

        Returns:
            Sequence[ToolExecutionSuggestion]: The tool execution suggestions
                from the first choice
        """
        if len(self.choices) > 1:
            logger.warning(
                "Choices list is > 1. Coudn't determine the tool calls. "
                + "Please, use the get_tool_calls method, instead, "
                + "passing the choice number you want to get the tool calls."
                + "Returning the first choice tool calls."
            )

        return self.get_tool_calls(0)

    @overload
    def clone[T_Schema](
        self,
        *,
        new_parseds: Sequence[T_Schema],
        new_id: uuid.UUID | None = None,
        new_object: Literal["chat.generation"] | None = None,
        new_created: datetime.datetime | None = None,
        new_model: str | None = None,
        new_choices: None = None,
        new_usage: Usage | None = None,
    ) -> Generation[T_Schema]: ...

    @overload
    def clone[T_Schema](
        self,
        *,
        new_parseds: None = None,
        new_id: uuid.UUID | None = None,
        new_object: Literal["chat.generation"] | None = None,
        new_created: datetime.datetime | None = None,
        new_model: str | None = None,
        new_choices: Sequence[Choice[T_Schema]],
        new_usage: Usage | None = None,
    ) -> Generation[T_Schema]: ...

    @overload
    def clone(
        self,
        *,
        # Nenhum destes é fornecido para este overload
        new_parseds: None = None,
        new_choices: None = None,
        new_id: uuid.UUID | None = None,
        new_object: Literal["chat.generation"] | None = None,
        new_created: datetime.datetime | None = None,
        new_model: str | None = None,
        new_usage: Usage | None = None,
    ) -> Generation[T]: ...  # Retorna o mesmo tipo T

    def clone[T_Schema](  # type: ignore[override]
        self,
        *,
        new_parseds: Sequence[T_Schema] | None = None,
        new_id: uuid.UUID | None = None,
        new_object: Literal["chat.generation"] | None = None,
        new_created: datetime.datetime | None = None,
        new_model: str | None = None,
        new_choices: Sequence[Choice[T_Schema]] | None = None,
        new_usage: Usage | None = None,
    ) -> Generation[T_Schema] | Generation[T]:  # Adjusted return type hint for clarity
        """
        Create a clone of this Generation, optionally with modified attributes.

        This method creates a new Generation object based on the current one,
        with the option to modify specific attributes. It supports several scenarios:

        1. Creating a new Generation with the same structure but different parsed data
        2. Creating a new Generation with entirely new choices
        3. Creating a simple clone with optional metadata changes

        The method uses overloads to provide proper type safety depending on which
        scenario is being used.

        Args:
            new_parseds: New parsed data to use in place of existing parsed data
            new_elapsed_time: New elapsed time value
            new_id: New ID for the generation
            new_object: New object type identifier
            new_created: New creation timestamp
            new_model: New model identifier
            new_choices: New choices to replace the existing ones
            new_usage: New usage statistics

        Returns:
            A new Generation object with the requested modifications

        Raises:
            ValueError: If both new_parseds and new_choices are provided, which
                would be ambiguous
        """
        # Validate against ambiguous parameter usage
        if new_choices and new_parseds:
            raise ValueError(
                "Cannot provide 'new_choices' together with 'new_parseds'."
            )

        # Scenario 1: Clone with new parsed data
        if new_parseds:
            # Validate length consistency
            if len(new_parseds) != len(self.choices):
                raise ValueError(
                    f"The number of 'new_parseds' ({len(new_parseds)}) does not match the number of existing 'choices' ({len(self.choices)})."
                )

            _new_choices_scenario1: list[Choice[T_Schema]] = [
                Choice(
                    message=GeneratedAssistantMessage(
                        # Use deepcopy for parts to ensure independence
                        parts=copy.deepcopy(choice.message.parts),
                        parsed=new_parseds[choice.index],
                    ),
                    index=choice.index,
                )
                for choice in self.choices
            ]

            return Generation[T_Schema](
                id=new_id or self.id,
                object=new_object or self.object,
                created=new_created or self.created,
                model=new_model or self.model,
                choices=_new_choices_scenario1,
                usage=(new_usage or self.usage).model_copy(),
            )

        # Scenario 2: Clone with entirely new choices provided
        if new_choices:
            return Generation[T_Schema](
                id=new_id or self.id,
                object=new_object or self.object,
                created=new_created or self.created,
                model=new_model or self.model,
                choices=new_choices,
                usage=(new_usage or self.usage).model_copy(),
            )

        # Scenario 3: Simple clone (same type T), potentially updating metadata
        if not new_parseds and not new_choices:
            # Deep copy existing choices to ensure independence
            _new_choices_scenario3: list[Choice[T]] = [
                copy.deepcopy(choice) for choice in self.choices
            ]
            # Cast is needed because the method signature expects T_Schema, but in this branch,
            # we know we are returning Generation[T]. Overloads handle the public API typing.
            return Generation[T](  # type: ignore[return-value]
                id=new_id or self.id,
                object=new_object or self.object,
                created=new_created or self.created,
                model=new_model or self.model,
                choices=_new_choices_scenario3,  # type: ignore[arg-type]
                usage=(new_usage or self.usage).model_copy(),
            )

        # Should be unreachable if overloads cover all valid cases and validation works
        raise ValueError(
            "Invalid combination of parameters for clone method. Use one of the defined overloads."
        )

    def tool_calls_amount(self) -> int:
        """
        Get the number of tool execution suggestions in the first choice.

        Returns:
            int: The number of tool execution suggestions
        """
        return len(self.tool_calls)

    def get_tool_calls(self, choice: int = 0) -> Sequence[ToolExecutionSuggestion]:
        """
        Get tool execution suggestions from a specific choice.

        Args:
            choice: The index of the choice to get tool calls from (default: 0)

        Returns:
            Sequence[ToolExecutionSuggestion]: The tool execution suggestions
                from the specified choice
        """
        return self.choices[choice].message.tool_calls

    @classmethod
    def mock(cls) -> Generation[T]:
        """
        Create a mock Generation object for testing purposes.

        This method creates a Generation with minimal default values,
        useful for testing without making actual API calls.

        Returns:
            Generation[T]: A mock Generation object
        """
        return cls(
            model="mock-model",
            id=uuid.uuid4(),
            object="chat.generation",
            created=datetime.datetime.now(),
            choices=[],
            usage=Usage(prompt_tokens=0, completion_tokens=0),
        )

    @property
    def text(self) -> str:
        """
        Get the concatenated text from all choices.

        This is a convenience property that returns all the text content
        from all choices concatenated into a single string.

        Returns:
            str: The concatenated text from all choices
        """
        return "".join([choice.message.text for choice in self.choices])

    def get_parsed(self, choice: int) -> T:
        """
        Get the parsed structured data from a specific choice.

        Args:
            choice: The index of the choice to get parsed data from

        Returns:
            T: The parsed structured data from the specified choice
        """
        return self.choices[choice].message.parsed


class ThoughtDetail(BaseModel):
    """
    A detailed explanation of a specific aspect of a reasoning step.

    ThoughtDetail represents the most granular level of reasoning in the Chain of Thought
    framework. Each detail captures a single observation, calculation, inference, or
    consideration that contributes to the reasoning step it belongs to.

    This granularity serves several purposes:
    - Allows precise examination of each atomic reasoning component
    - Makes complex reasoning steps more digestible through decomposition
    - Provides clear traceability of how each consideration affects the reasoning
    - Enables targeted feedback or correction at the most specific level

    Think of ThoughtDetails as the "atomic units" of a reasoning process that,
    when combined, form Steps, which in turn form a complete Chain of Thought.

    Attributes:
        detail: A granular explanation of a specific aspect of the reasoning step

    Example:
        >>> ThoughtDetail(detail="First, I added 2 + 3")
        >>> ThoughtDetail(detail="The velocity must be constant because acceleration is zero")
    """

    detail: str = Field(
        description="A granular explanation of a specific aspect of the reasoning step.",
        # examples=["First, I added 2 + 3", "Checked if the number is even or odd"],
    )


class Step(BaseModel):
    """
    A single step in a chain of thought reasoning process.

    Each Step represents a distinct phase in a logical reasoning sequence,
    containing both a high-level explanation of what was done in this step
    and a collection of more granular details that elaborate on specific
    aspects of the reasoning.

    Steps are numbered to maintain a clear sequence of the reasoning process,
    allowing for proper ordering when presented to users or when analyzing
    the reasoning path.

    In complex reasoning tasks, having explicit steps helps in:
    - Breaking down complex problems into manageable parts
    - Identifying exactly where reasoning might go astray
    - Providing visibility into the full logical progression
    - Making the overall reasoning process more understandable

    Attributes:
        step_number: The position of this step in the overall chain of thought
        explanation: A concise description of what was done in this step
        details: A list of specific details for each step in the reasoning

    Example:
        >>> Step(
        ...     step_number=1,
        ...     explanation="Analyze the input statement",
        ...     details=[
        ...         ThoughtDetail(detail="Check initial values"),
        ...         ThoughtDetail(detail="Confirm there are no inconsistencies")
        ...     ]
        ... )
    """

    step_number: int = Field(
        description="The position of this step in the overall chain of thought.",
    )

    explanation: str = Field(
        description="A concise description of what was done in this step.",
    )

    details: Sequence[ThoughtDetail] = Field(
        description="A list of specific details for each step in the reasoning.",
    )


class ChainOfThought[T](BaseModel):
    """
    Structured reasoning process with final answer.

    This class represents a complete chain-of-thought reasoning process,
    breaking down complex problem-solving into a sequence of explicit steps
    with detailed explanations, culminating in a final answer or conclusion.

    Chain of Thought is particularly useful for:
    - Complex reasoning tasks requiring step-by-step thinking
    - Making model reasoning transparent and verifiable
    - Debugging or explaining how a model arrived at a conclusion
    - Implementing research techniques like Chain of Thought prompting

    The generic type parameter T allows the final_answer to be of any type,
    such as a string, number, boolean, or complex structured data.

    Attributes:
        general_title: High-level description of reasoning goal
        steps: Logical steps in reasoning process
        final_answer: Conclusion of the reasoning chain, of type T

    Example:
        >>> ChainOfThought(
        ...     general_title="Math problem solution",
        ...     steps=[step1, step2],
        ...     final_answer=42
        ... )
    """

    general_title: str = Field(
        description="A brief label or description that identifies the purpose of the reasoning.",
        # examples=["Sum of two numbers", "Logical problem solving"],
    )

    steps: Sequence[Step] = Field(
        description="The sequence of steps that make up the full reasoning process.",
    )

    final_answer: T = Field(
        description="The conclusion or result after all the reasoning steps."
    )

    def as_string(self, lang: str = "en") -> str:
        """Return a localized string representation of the ChainOfThought.

        Args:
            lang: ISO language code for the output format (default: "en" for English)

        Returns:
            str: A formatted string showing the reasoning process and final answer in the specified language

        Example:
            >>> print(chain_of_thought.as_string())  # Default English
            MATH PROBLEM SOLUTION

            Step 1: Analyze input data
            - Data: 234 and 567
            - Check if they are integers

            Step 2: Perform the calculation
            - 234 + 567 = 801

            Final Answer: 801

            >>> print(chain_of_thought.as_string("es"))  # Spanish
            SOLUCIÓN DEL PROBLEMA MATEMÁTICO

            Paso 1: Analizar datos de entrada
            - Datos: 234 y 567
            - Comprobar si son números enteros

            Paso 2: Realizar el cálculo
            - 234 + 567 = 801

            Respuesta Final: 801

            >>> print(chain_of_thought.as_string("pt"))  # Portuguese
            SOLUÇÃO DO PROBLEMA MATEMÁTICO

            Passo 1: Analisar dados de entrada
            - Dados: 234 e 567
            - Verificar se são números inteiros

            Passo 2: Realizar o cálculo
            - 234 + 567 = 801

            Resposta Final: 801
        """
        # Define language-specific terms
        translations = {
            "en": {"step": "Step", "final_answer": "Final Answer"},
            "es": {"step": "Paso", "final_answer": "Respuesta Final"},
            "fr": {"step": "Étape", "final_answer": "Réponse Finale"},
            "de": {"step": "Schritt", "final_answer": "Endgültige Antwort"},
            "pt": {"step": "Passo", "final_answer": "Resposta Final"},
        }

        # Default to English if requested language is not available
        if lang not in translations:
            lang = "en"

        # Get translation dictionary for the specified language
        t = translations[lang]

        # Start with the title
        result = f"{self.general_title.upper()}\n\n"

        # Add each step with its details
        for step in self.steps:
            result += f"{t['step']} {step.step_number}: {step.explanation}\n"
            for detail in step.details:
                result += f"- {detail.detail}\n"
            result += "\n"

        # Add the final answer
        result += f"{t['final_answer']}: {self.final_answer}"

        return result


class AudioDescription(BaseModel):
    """Detailed description of audio content.

    Attributes:
        overall_description: Comprehensive content summary
        content_type: Category (podcast, music, etc.)
        audio_elements: Individual components
        structure: Spatial organization
        dominant_auditory_features: Salient auditory characteristics
        intended_purpose: Interpreted purpose

    Example:
        >>> desc = AudioDescription(
        ...     content_type="podcast",
        ...     audio_elements=[...]
        ... )
    """

    overall_description: str = Field(
        title="Overall Audio Description",
        description="Provide a comprehensive and detailed narrative describing the entire audio media, focusing on its content, structure, and key auditory elements. Imagine you are explaining the audio to someone who cannot hear it. Describe the overall purpose or what information the audio is conveying or what experience it aims to create. Detail the main components and how they are organized. Use precise language to describe auditory characteristics like pitch, tone, rhythm, tempo, and instrumentation. For abstract audio, focus on describing the sonic properties and composition. Think about the key aspects someone needs to understand to grasp the content and structure of the audio. Examples: 'The audio presents a news report detailing recent events, featuring a clear and professional narration with background music.', 'The audio is a piece of ambient music featuring layered synthesizers and natural soundscapes, creating a calming atmosphere.', 'The audio recording captures a lively conversation between two individuals, with distinct voices and occasional laughter.'",
        examples=[
            "A podcast discussing current events",
            "A musical piece with a strong melody",
            "A recording of nature sounds",
        ],
    )


class AudioElementDescription(BaseModel):
    """A description of an audio element within a media file.

    Attributes:
        type: The general category of this audio element within the media
        details: A detailed description of the auditory characteristics and properties of this element
        role: The purpose or function of this audio element within the context of the media
        relationships: How this audio element is related to other elements in the media

    Example:
        >>> AudioElementDescription(
        ...     type="Speech segment",
        ...     details="The word 'example' spoken with emphasis",
        ...     role="Introduces the main subject",
        ...     relationships=["Occurs after a period of silence"]
        ... )
    """

    type: str = Field(
        title="Element Type",
        description="The general category of this audio element within the media. This could be a type of sound, a segment of speech, a musical phrase, or any other distinct auditory component. Examples: 'Speech segment', 'Musical note', 'Sound effect', 'Silence', 'Jingle'.",
        examples=[
            "Speech",
            "Melody",
            "Footsteps",
            "Silence",
        ],
    )

    details: str = Field(
        title="Element Details",
        description="A detailed description of the auditory characteristics and properties of this element. For speech, provide the content or a description of the speaker's tone and delivery. For music, describe the melody, harmony, rhythm, and instrumentation. For sound effects, describe the sound and its characteristics. Be specific and descriptive. Examples: 'The spoken phrase 'Hello world' in a clear voice', 'A high-pitched sustained note on a violin', 'The sound of a door slamming shut', 'A brief period of complete silence'.",
        examples=[
            "The word 'example' spoken with emphasis",
            "A low humming sound",
            "A sharp, percussive beat",
        ],
    )

    role: str | None = Field(
        default=None,
        title="Element Role/Function",
        description="The purpose or function of this audio element within the context of the media. How does it contribute to the overall meaning, mood, or structure? For example, in a song, describe its role in the melody or harmony. In a spoken piece, explain its informational or emotional contribution. In a soundscape, its contribution to the atmosphere. Examples: 'Conveys information about the topic', 'Creates a sense of tension', 'Marks the beginning of a new section', 'Provides background ambience'.",
        examples=[
            "Introduces the main subject",
            "Builds suspense",
            "Signals a transition",
            "Establishes the setting",
        ],
    )

    relationships: Sequence[str] | None = Field(
        default=None,
        title="Element Relationships",
        description="Describe how this audio element is related to other elements in the media. Explain its temporal relationship to others, whether it occurs before, during, or after other sounds, or how it interacts with other auditory elements. Be specific about the other elements involved. Examples: 'This musical phrase follows the introductory melody', 'The sound effect occurs simultaneously with the visual impact', 'The speaker's voice overlaps with the background music'.",
        examples=[
            "Occurs after a period of silence",
            "Plays under the main narration",
            "A response to the previous sound",
        ],
    )

    def md(self, indent_level: int = 0) -> str:
        indent = "  " * indent_level
        md_str = f"{indent}**Element Type**: {self.type}\n"
        md_str += f"{indent}**Element Details**: {self.details}\n"
        if self.role:
            md_str += f"{indent}**Role/Function**: {self.role}\n"
        if self.relationships:
            md_str += f"{indent}**Relationships**:\n"
            for rel in self.relationships:
                md_str += f"{indent}  - {rel}\n"
        return md_str


class AudioStructure(BaseModel):
    """A description of the overall structure and organization of an audio media file.

    Attributes:
        organization: A description of how the audio elements are arranged and organized within the media
        groupings: Significant groupings of elements that appear to function together
        focal_point: The primary focal point that draws attention

    Example:
        >>> AudioStructure(
        ...     organization="A narrative with a clear beginning, middle, and end",
        ...     groupings=["The introduction of the song", "The main argument of the speech"],
        ...     focal_point="The main theme of the music"
        ... )
    """

    organization: str | None = Field(
        default=None,
        title="Overall Organization",
        description="A description of how the audio elements are arranged and organized within the media. Describe the overall structure, flow, or pattern. Is it linear, cyclical, thematic, or something else? How are the different parts connected or separated? Examples: 'A song with verse-chorus structure', 'A chronological sequence of spoken events', 'A layered soundscape with overlapping elements'.",
        examples=[
            "A narrative with a clear beginning, middle, and end",
            "A repeating musical motif",
            "A conversation with alternating speakers",
        ],
    )

    groupings: Sequence[str] | None = Field(
        default=None,
        title="Significant Groupings of Elements",
        description="Describe any notable groupings or clusters of audio elements that appear to function together or have a shared context. Explain what binds these elements together aurally or conceptually. Examples: 'The instrumental section of the song', 'A dialogue between two characters', 'A series of related sound effects'.",
        examples=[
            "The introduction of the song",
            "The main argument of the speech",
            "The sounds of a busy street",
        ],
    )

    focal_point: str | None = Field(
        default=None,
        title="Primary Focal Point",
        description="Identify the most prominent or central audio element or section that draws the listener's attention. Explain why this element stands out (e.g., volume, pitch, prominence of a voice or instrument). If there isn't a clear focal point, describe the distribution of auditory emphasis. Examples: 'The lead vocalist's melody', 'The loudest sound effect', 'The central argument of the speech'.",
        examples=[
            "The main theme of the music",
            "The key statement in the narration",
            "A sudden loud bang",
        ],
    )

    def md(self, indent_level: int = 1) -> str:
        indent = "  " * indent_level
        md_str = ""
        if self.organization:
            md_str += f"{indent}**Overall Organization**: {self.organization}\n"
        if self.groupings:
            md_str += f"{indent}**Significant Groupings of Elements**:\n"
            for group in self.groupings:
                md_str += f"{indent}  - {group}\n"
        if self.focal_point:
            md_str += f"{indent}**Primary Focal Point**: {self.focal_point}\n"
        return md_str


class GraphicalElementDescription(BaseModel):
    """A description of a visual or auditory element within a media file.

    Attributes:
        type: The general category of this visual element within the media
        details: A detailed description of the characteristics and properties of this element
        role: The purpose or function of this element within the context of the media
        relationships: How this element is related to other elements in the media

    Example:
        >>> GraphicalElementDescription(
        ...     type="Text string",
        ...     details="The number '3' in the top-left corner",
        ...     role="Represents the coefficient of x",
        ...     relationships=["Located above the main equation"]
        ... )
    """

    type: str = Field(
        title="Element Type",
        description="The general category of this visual element within the media. This could be a recognizable object, a symbol, a graphical component, a section of text, or any other distinct visual or temporal component. Be descriptive but not necessarily tied to real-world objects if the media is abstract or symbolic. Examples: 'Equation term', 'Geometric shape', 'Timeline marker', 'Audio waveform segment', 'Brushstroke', 'Data point'.",
        examples=[
            "Text string",
            "Geometric shape",
            "Timeline marker",
            "Component of a machine",
            "Area of color",
            "Video transition",
        ],
    )

    details: str = Field(
        title="Element Details",
        description="A detailed description of the characteristics and properties of this element. Focus on what is visually or audibly apparent. For text, provide the content. For shapes, describe form, color, and features. For abstract elements, describe visual properties like color, texture, and form, or temporal properties like duration and transitions. Be specific and descriptive. Examples: 'The text string 'y = mx + c' in bold font', 'A red circle with a thick black outline', 'A sudden fade to black', 'A high-pitched tone'.",
        examples=[
            "The number '3' in the top-left corner",
            "A thin, dashed black line",
            "A vibrant green triangular shape",
            "A slow zoom-in effect",
            "A burst of static",
        ],
    )

    role: str | None = Field(
        default=None,
        title="Element Role/Function",
        description="The purpose or function of this element within the context of the media. How does it contribute to the overall meaning, structure, or flow? For example, in a formula, describe its mathematical role. In a diagram, its function. In a video, its narrative or informational contribution. Examples: 'Represents a variable in the equation', 'Indicates the direction of flow', 'Marks a key event in the timeline', 'Signals a change in scene'.",
        examples=[
            "Represents the coefficient of x",
            "Connects two stages in the process",
            "Highlights a critical moment",
            "Provides context for the following scene",
        ],
    )

    relationships: Sequence[str] | None = Field(
        default=None,
        title="Element Relationships",
        description="Describe how this element is related to other elements in the media. "
        + "Explain its position relative to others, whether it's connected, overlapping, "
        + "near, or otherwise associated with them, considering spatial and temporal "
        + "relationships. Be specific about the other elements involved. Examples: "
        + "'The arrow points from this box to the next', 'This circle is enclosed"
        + "within the square', 'This scene follows the previous one', 'The music"
        + "swells during this visual element'.",
        examples=[
            "Located above the main equation",
            "Connected to the previous step by a line",
            "Part of a larger assembly",
            "Occurs immediately after the title card",
        ],
    )

    extracted_text: str | None = Field(
        default=None,
        title="Extracted Text Content",
        description="For elements that contains text elements, the actual textual content "
        + "extracted through OCR. Preserves line breaks and spatial relationships where "
        + "possible.",
        examples=["'3.14'", "'Warning: Do not open'", "'y = mx + b'"],
    )

    def md(self, indent_level: int = 0) -> str:
        indent = "  " * indent_level
        md_str = f"{indent}**Element Type**: {self.type}\n"
        md_str += f"{indent}**Element Details**: {self.details}\n"
        if self.role:
            md_str += f"{indent}**Role/Function**: {self.role}\n"
        if self.relationships:
            md_str += f"{indent}**Relationships**:\n"
            for rel in self.relationships:
                md_str += f"{indent}  - {rel}\n"
        return md_str


class MediaStructure(BaseModel):
    """A description of the overall structure and organization of a visual or auditory media file.

    Attributes:
        layout: A description of how the elements are arranged and organized within the media
        groupings: Significant groupings or clusters of elements that appear to function together
        focal_point: The primary focal point that draws attention

    Example:
        >>> MediaStructure(
        ...     layout="A step-by-step diagram",
        ...     groupings=["The main body of the text", "The elements forming the control panel"],
        ...     focal_point="The large heading at the top"
        ... )
    """

    layout: str | None = Field(
        default=None,
        title="Overall Layout and Organization",
        description="A description of how the elements are arranged and organized within the media. Describe the overall structure, flow, or pattern, considering both spatial and temporal aspects. Is it linear, grid-based, hierarchical, sequential, or something else? How are the different parts connected or separated? Examples: 'A top-down flowchart', 'A grid of data points', 'A chronological sequence of scenes', 'A central diagram with surrounding labels'.",
        examples=[
            "A step-by-step diagram",
            "A clustered arrangement of shapes",
            "A formula presented on a single line",
            "A narrative with distinct acts",
        ],
    )

    groupings: Sequence[str] | None = Field(
        default=None,
        title="Significant Groupings of Elements",
        description="Describe any notable groupings or clusters of elements that appear to function together or have a shared context, considering both visual and temporal coherence. Explain what binds these elements together visually, aurally, or conceptually. Examples: 'The terms on the left side of the equation', 'The interconnected components of the circuit diagram', 'A montage of related images', 'A musical theme associated with a character'.",
        examples=[
            "The main body of the text",
            "The elements forming the control panel",
            "The interconnected nodes of the network",
            "A series of shots depicting the same event",
        ],
    )

    focal_point: str | None = Field(
        default=None,
        title="Primary Focal Point",
        description="Identify the most prominent or central element or area that draws attention, considering visual, auditory, and temporal emphasis. Explain why this element stands out (e.g., size, color, position, duration, sound intensity). If there isn't a clear focal point, describe the distribution of emphasis. Examples: 'The main title of the document', 'The central component of the machine', 'The climax of the scene', 'The loudest sound'.",
        examples=[
            "The large heading at the top",
            "The brightly colored area in the center",
            "The main subject of the drawing",
            "The key moment of impact",
        ],
    )

    def md(self, indent_level: int = 1) -> str:
        indent = "  " * indent_level
        md_str = ""
        if self.layout:
            md_str += f"{indent}**Overall Layout and Organization**: {self.layout}\n"
        if self.groupings:
            md_str += f"{indent}**Significant Groupings of Elements**:\n"
            for group in self.groupings:
                md_str += f"{indent}  - {group}\n"
        if self.focal_point:
            md_str += f"{indent}**Primary Focal Point**: {self.focal_point}\n"
        return md_str


class QueryExpansion(BaseModel):
    """
    Represents an expanded version of a search query.

    This class can contain an expanded query string that adds context,
    specificity, or alternative phrasings to the original query.
    """

    expanded_query: str | None = Field(
        default=None,
        description="The expanded query string or None if expansion is not possible or needed.",
    )


class VisualMediaDescription(BaseModel):
    """Detailed description of visual content.

    Attributes:
        overall_description: Comprehensive content summary
        content_type: Category (diagram, photo, etc.)
        visual_elements: Individual components
        structure: Spatial organization
        dominant_features: Salient visual characteristics
        intended_purpose: Interpreted purpose

    Example:
        >>> desc = VisualMediaDescription(
        ...     content_type="infographic",
        ...     visual_elements=[...]
        ... )
    """

    overall_description: str = Field(
        title="Overall Media Description",
        description="Provide a comprehensive and detailed narrative describing the entire visual media, focusing on its content, structure, and key elements. Imagine you are explaining it to someone who cannot see or hear it. Describe the overall purpose or what information it is conveying. Detail the main components and how they are organized, considering both spatial and temporal aspects. Use precise language to describe visual characteristics like shapes, colors, patterns, and relationships, as well as temporal characteristics like duration, transitions, and pacing. For abstract media, focus on describing the properties and composition. Think about the key aspects someone needs to understand to grasp the content and structure. Examples: 'The video presents a step-by-step tutorial on assembling a device. Text overlays accompany the visual demonstrations.', 'The animated graphic shows the flow of data through a network, with arrows indicating direction and color-coding representing different types of data.', 'The abstract animation features pulsating colors and evolving geometric shapes set to a rhythmic soundtrack.'",
        examples=[
            "A diagram illustrating the water cycle",
            "A complex algebraic equation",
            "An abstract painting with bold colors",
            "A short film depicting a historical event",
        ],
    )

    content_type: str = Field(
        title="Content Type",
        description="A general categorization of the audio's content. This helps to broadly define what kind of auditory experience or information is being presented. Examples: 'Podcast', 'Song', 'Speech', 'Sound effects', 'Ambient music', 'Audiobook', 'Interview'.",
        examples=["Podcast", "Music", "Speech", "Sound Effects"],
    )

    audio_elements: Sequence[AudioElementDescription] | None = Field(
        default=None,
        title="Detailed Audio Element Descriptions",
        description="A list of individual audio elements identified within the media, each with its own detailed description. For each element, provide its type, specific auditory details, its role or function within the audio's context, and its relationships to other elements. The goal is to break down the audio into its fundamental auditory components and describe them comprehensively. This applies to all types of audio, from spoken words in a podcast to musical notes in a song or distinct sound effects.",
    )

    visual_elements: Sequence[GraphicalElementDescription] | None = Field(
        default=None, description="Visual elements"
    )

    dominant_features: Sequence[str] | None = Field(
        default=None, description="Dominant features of the Visual media"
    )

    structure: AudioStructure | None = Field(
        default=None,
        title="Audio Structure and Organization",
        description="A description of the overall structure and organization of the audio elements within the media. This section focuses on how the different parts are arranged and related to each other. Describe the overall organization, any significant groupings of elements, and the primary focal point or area of emphasis. This helps to understand the higher-level organization of the audio's content.",
    )

    dominant_auditory_features: Sequence[str] | None = Field(
        default=None,
        title="Dominant Auditory Features",
        description="A list of the most striking auditory features of the audio that contribute significantly to its overall character and impact. This could include dominant melodies, rhythmic patterns, distinctive voices or timbres, recurring sound effects, or any other salient auditory characteristics. Be specific and descriptive. Examples: 'A strong, repetitive beat', 'A high-pitched, clear female voice', 'Frequent use of echo and reverb', 'A melancholic piano melody'.",
        examples=[
            "A fast tempo",
            "A deep bassline",
            "Clear and articulate speech",
        ],
    )

    intended_purpose: str | None = Field(
        default=None,
        title="Intended Purpose or Meaning",
        description="An interpretation of the intended purpose or meaning of the audio, based on its content and structure. What is the audio trying to convey or communicate? For a song, it might be to express emotions. For a podcast, to inform or entertain. For sound effects, to create a specific atmosphere. This is an interpretive field, so focus on reasonable inferences based on the auditory evidence. Examples: 'To tell a story through sound', 'To provide information on a specific topic', 'To create a relaxing and immersive soundscape', 'To evoke feelings of joy and excitement'.",
        examples=[
            "To entertain the listener",
            "To educate on a particular subject",
            "To create a sense of atmosphere",
        ],
    )

    @property
    def md(self) -> str:
        md_str = f"## Overall Media Description\n{self.overall_description}\n\n"
        md_str += f"## Content Type\n{self.content_type}\n\n"

        if self.visual_elements:
            md_str += "## Detailed Element Descriptions\n"
            for element in self.visual_elements:
                md_str += element.md(indent_level=0) + "\n"

        if self.structure:
            md_str += "## Media Structure and Organization\n"
            md_str += self.structure.md(indent_level=1) + "\n"

        if self.dominant_features:
            md_str += "## Dominant Features\n"
            for feature in self.dominant_features:
                md_str += f"- {feature}\n"
            md_str += "\n"

        if self.intended_purpose:
            md_str += f"## Intended Purpose or Meaning\n{self.intended_purpose}\n\n"

        return md_str
