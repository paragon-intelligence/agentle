"""
Tool module for creating and managing function-callable tools in Agentle.

This module provides the core Tool class used throughout the Agentle framework to represent
callable functions as tools that can be used by AI models. Tools are a fundamental building
block in the framework that enable AI agents to interact with external systems, retrieve
information, and perform actions in the real world.

The Tool class encapsulates a callable function along with metadata such as name, description,
and parameter specifications. It can be created either directly from a callable Python function
or by converting from MCP (Model Control Protocol) tool format.

Tools are typically used in conjunction with Agents to provide them with capabilities to
perform specific tasks. When an Agent decides to use a tool, it provides the necessary arguments,
and the Tool executes the underlying function with those arguments.

Example:
```python
from agentle.generations.tools.tool import Tool

# Create a tool from a function
def get_weather(location: str, unit: str = "celsius") -> str:
    \"\"\"Get current weather for a location\"\"\"
    # Implementation would typically call a weather API
    return f"The weather in {location} is sunny. Temperature is 25°{unit[0].upper()}"

# Create a tool instance from the function
weather_tool = Tool.from_callable(get_weather)

# Use the tool directly
result = weather_tool.call(location="Tokyo", unit="celsius")
print(result)  # "The weather in Tokyo is sunny. Temperature is 25°C"
```
"""

from __future__ import annotations

import base64
import inspect
from collections.abc import Awaitable, Callable, MutableSequence
from typing import TYPE_CHECKING, Any, Literal

from rsb.coroutines.run_sync import run_sync
from rsb.models.base_model import BaseModel
from rsb.models.config_dict import ConfigDict
from rsb.models.field import Field
from rsb.models.private_attr import PrivateAttr

from agentle.generations.models.message_parts.file import FilePart
from agentle.mcp.servers.mcp_server_protocol import MCPServerProtocol

if TYPE_CHECKING:
    from mcp.types import Tool as MCPTool
    from agentle.agents.context import Context


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

    # change to private
    _before_call: Callable[..., T_Output] | None = PrivateAttr(
        default=None,
    )

    _after_call: Callable[..., T_Output] | None = PrivateAttr(
        default=None,
    )

    _callable_ref: (
        Callable[..., T_Output] | Callable[..., Awaitable[T_Output]] | None
    ) = PrivateAttr(default=None)

    _server: MCPServerProtocol | None = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    def is_mcp_tool(self) -> bool:
        return self._server is not None

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

    def call(self, context: Context | None = None, **kwargs: object) -> T_Output:
        """
        Executes the underlying function with the provided arguments.

        This method calls the function referenced by the `_callable_ref` attribute
        with the provided keyword arguments. It raises a ValueError if the Tool
        was not created with a callable reference.

        Args:
            context: Optional context object for HITL workflows and state management.
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
        ret = run_sync(self.call_async, timeout=None, context=context, **kwargs)
        return ret

    async def call_async(
        self, context: Context | None = None, **kwargs: object
    ) -> T_Output:
        """
        Executes the underlying function asynchronously with the provided arguments.

        This method calls the function referenced by the `_callable_ref` attribute
        with the provided keyword arguments. It raises a ValueError if the Tool
        was not created with a callable reference.

        Args:
            context: Optional context object for HITL workflows and state management.
            **kwargs: Keyword arguments to pass to the underlying function.

        Returns:
            T_Output: The result of calling the underlying function.

        Raises:
            ValueError: If the Tool does not have a callable reference.

        Example:
            ```python
            async def async_add(a: int, b: int) -> int:
                \"\"\"Add two numbers\"\"\"
                return a + b

            add_tool = Tool.from_callable(async_add)
            result = await add_tool.call_async(a=5, b=3)
            print(result)  # Output: 8
            ```
        """
        if self._callable_ref is None:
            raise ValueError(
                'Tool is not callable because the "_callable_ref" instance variable is not set'
            )

        # Execute before_call callback with context if available
        if self._before_call is not None:
            if inspect.iscoroutinefunction(self._before_call):
                if context is not None:
                    await self._before_call(context=context, **kwargs)
                else:
                    await self._before_call(**kwargs)
            else:
                if context is not None:
                    self._before_call(context=context, **kwargs)
                else:
                    self._before_call(**kwargs)

        # Execute the main function
        if inspect.iscoroutinefunction(self._callable_ref):
            ret: T_Output = await self._callable_ref(**kwargs)  # type: ignore
        else:
            ret: T_Output = self._callable_ref(**kwargs)  # type: ignore

        # Execute after_call callback with context and result if available
        if self._after_call is not None:
            if inspect.iscoroutinefunction(self._after_call):
                if context is not None:
                    await self._after_call(context=context, result=ret, **kwargs)
                else:
                    await self._after_call(result=ret, **kwargs)
            else:
                if context is not None:
                    self._after_call(context=context, result=ret, **kwargs)
                else:
                    self._after_call(result=ret, **kwargs)

        return ret

    @classmethod
    def from_mcp_tool(
        cls, mcp_tool: MCPTool, server: MCPServerProtocol
    ) -> Tool[T_Output]:
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
        from mcp.types import (
            BlobResourceContents,
            CallToolResult,
            EmbeddedResource,
            ImageContent,
            TextContent,
            TextResourceContents,
        )

        tool = cls(
            name=mcp_tool.name,
            description=mcp_tool.description,
            parameters=mcp_tool.inputSchema,
        )
        tool._server = server

        async def _callable_ref(**kwargs: object) -> Any:
            call_tool_result: CallToolResult = await server.call_tool_async(
                tool_name=mcp_tool.name,
                arguments=kwargs,
            )

            contents: MutableSequence[str | FilePart] = []

            for content in call_tool_result.content:
                match content:
                    case TextContent():
                        contents.append(content.text)
                    case ImageContent():
                        contents.append(
                            FilePart(
                                data=base64.b64decode(content.data),
                                mime_type=content.mimeType,
                            )
                        )
                    case EmbeddedResource():
                        match content.resource:
                            case TextResourceContents():
                                contents.append(content.resource.text)
                            case BlobResourceContents():
                                contents.append(
                                    FilePart(
                                        data=base64.b64decode(content.resource.blob),
                                        mime_type="application/octet-stream",
                                    )
                                )

            return contents

        tool._callable_ref = _callable_ref
        return tool

    @classmethod
    def from_callable(
        cls,
        _callable: Callable[..., T_Output],
        before_call: Callable[..., T_Output] | None = None,
        after_call: Callable[..., T_Output] | None = None,
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
        instance._before_call = before_call
        instance._after_call = after_call

        return instance

    def __str__(self) -> str:
        return self.text
