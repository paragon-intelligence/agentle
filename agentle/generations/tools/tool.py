"""
Tool module for creating and managing function-callable tools in Agentle.

This module provides the core Tool class used throughout the Agentle framework to represent
callable functions as tools that can be used by AI models. Tools are a fundamental building
block in the framework that enable AI agents to interact with external systems, retrieve
information, and perform actions in the real world.

The Tool class encapsulates a callable function along with metadata such as name, description,
and parameter specifications. It can be created either directly from a callable Python function
or by converting from MCP (Model Control Protocol) tool format.

Updated to use improved typing with better callback support.
"""

from __future__ import annotations

import base64
import inspect
from collections.abc import Awaitable, Callable, MutableSequence
import logging
from typing import TYPE_CHECKING, Any, Literal, ParamSpec, TypeVar, Self

from rsb.coroutines.run_sync import run_sync
from rsb.models.base_model import BaseModel
from rsb.models.config_dict import ConfigDict
from rsb.models.field import Field
from rsb.models.private_attr import PrivateAttr

from agentle.generations.models.message_parts.file import FilePart
from agentle.mcp.servers.mcp_server_protocol import MCPServerProtocol

if TYPE_CHECKING:
    from mcp.types import Tool as MCPTool

_logger = logging.getLogger(__name__)

# Type variables for the from_callable method
CallableP = ParamSpec("CallableP")
CallableT = TypeVar("CallableT")


class Tool[**P = ..., T_Output = Any](BaseModel):
    """
    A callable tool that can be used by AI models to perform specific functions.

    The Tool class represents a callable function with associated metadata such as name,
    description, and parameter specifications. Tools are the primary mechanism for enabling
    AI agents to interact with external systems, retrieve information, and perform actions.

    A Tool instance can be created either directly from a Python callable function using the
    `from_callable` class method, or from an MCP (Model Control Protocol) tool format using
    the `from_mcp_tool` class method.

    The class is generic with T_Output representing the return type of the underlying callable function.
    When used with `from_callable`, the Tool preserves both parameter and return type information
    for full type safety.

    Type Parameters:
        P: ParamSpec for the callable's parameters
        T_Output: Return type of the callable function

    Attributes:
        type: Literal field that identifies this as a tool, always set to "tool".
        name: Human-readable name of the tool.
        description: Human-readable description of what the tool does.
        parameters: Dictionary of parameter specifications for the tool.
        _callable_ref: Private attribute storing the callable function.
        _before_call: Optional callback executed before the main function.
        _after_call: Optional callback executed after the main function.

    Examples:
        ```python
        # Create a tool from a function with full type safety
        def add_numbers(a: int, b: int) -> int:
            \"\"\"Add two numbers together\"\"\"
            return a + b

        add_tool = Tool.from_callable(add_numbers)
        # Now call is fully typed: (a: int, b: int) -> int
        result = add_tool.call(a=5, b=3)  # Type-safe call
        assert result == 8
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

    ignore_errors: bool = Field(
        default=False,
        description="If True, errors in the tool execution will be ignored and the agent will continue running.",
    )

    # Callable reference - using ParamSpec for better parameter type safety
    _callable_ref: Callable[P, T_Output] | Callable[P, Awaitable[T_Output]] | None = (
        PrivateAttr(default=None)
    )

    # Before call receives same params as main function and optionally return result to short-circuit
    _before_call: (
        Callable[P, T_Output | None] | Callable[P, Awaitable[T_Output | None]] | None
    ) = PrivateAttr(default=None)

    # After call receives result + original params and can modify the result
    _after_call: Callable[..., T_Output] | Callable[..., Awaitable[T_Output]] | None = (
        PrivateAttr(default=None)
    )

    _server: MCPServerProtocol | None = PrivateAttr(default=None)

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=False)

    def is_mcp_tool(self) -> bool:
        return self._server is not None

    @property
    def text(self) -> str:
        """
        Generates a human-readable text representation of the tool.

        Returns:
            str: A formatted string containing the tool name, description, and parameters.
        """
        return f"Tool: {self.name}\nDescription: {self.description}\nParameters: {self.parameters}"

    def call(self, *args: P.args, **kwargs: P.kwargs) -> T_Output:
        """
        Executes the underlying function with the provided arguments.

        Args:
            *args: Positional arguments matching the ParamSpec P of the underlying function.
            **kwargs: Keyword arguments matching the ParamSpec P of the underlying function.

        Returns:
            T_Output: The result of calling the underlying function.

        Raises:
            ValueError: If the Tool does not have a callable reference.
        """
        ret = run_sync(self.call_async, timeout=None, *args, **kwargs)
        return ret

    async def call_async(self, *args: P.args, **kwargs: P.kwargs) -> T_Output:
        """
        Executes the underlying function asynchronously with the provided arguments.

        Args:
            *args: Positional arguments matching the ParamSpec P of the underlying function.
            **kwargs: Keyword arguments matching the ParamSpec P of the underlying function.

        Returns:
            T_Output: The result of calling the underlying function.

        Raises:
            ValueError: If the Tool does not have a callable reference.
        """
        _logger.debug(
            f"Calling tool '{self.name}' with arguments: args={args}, kwargs={kwargs}"
        )

        if self._callable_ref is None:
            _logger.error(f"Tool '{self.name}' is not callable - missing _callable_ref")
            raise ValueError(
                'Tool is not callable because the "_callable_ref" instance variable is not set'
            )

        try:
            # Execute before_call callback - can short-circuit execution
            if self._before_call is not None:
                _logger.debug(f"Executing before_call callback for tool '{self.name}'")
                if inspect.iscoroutinefunction(self._before_call):
                    before_result = await self._before_call(*args, **kwargs)
                else:
                    before_result = self._before_call(*args, **kwargs)

                # If before_call returns a result, use it and skip main function
                if before_result is not None:
                    _logger.debug("before_call returned result, skipping main function")
                    return before_result  # type: ignore[return-value]

            # Execute the main function
            _logger.debug(f"Executing main function for tool '{self.name}'")
            if inspect.iscoroutinefunction(self._callable_ref):
                try:
                    async_ret: T_Output = await self._callable_ref(*args, **kwargs)
                except Exception as e:
                    if self.ignore_errors:
                        _logger.error(
                            f"Error executing tool '{self.name}': {str(e)}",
                            exc_info=True,
                        )
                        return f"Error while executing tool {self.name}: {str(e)}"  # type: ignore
                    else:
                        raise
                ret = async_ret
            else:
                try:
                    sync_ret: T_Output = self._callable_ref(*args, **kwargs)  # type: ignore[misc]
                except Exception as e:
                    if self.ignore_errors:
                        _logger.error(
                            f"Error executing tool '{self.name}': {str(e)}",
                            exc_info=True,
                        )
                        return f"Error while executing tool {self.name}: {str(e)}"  # type: ignore
                    else:
                        raise
                ret = sync_ret

            _logger.info(f"Tool '{self.name}' executed successfully")

            # Execute after_call callback - can modify the result
            if self._after_call is not None:
                _logger.debug(f"Executing after_call callback for tool '{self.name}'")
                if inspect.iscoroutinefunction(self._after_call):
                    # Pass result as first positional arg, then original args and kwargs
                    modified_result = await self._after_call(ret, *args, **kwargs)
                else:
                    modified_result = self._after_call(ret, *args, **kwargs)

                return modified_result  # type: ignore[return-value]

            return ret

        except Exception as e:
            _logger.error(
                f"Error executing tool '{self.name}': {str(e)}", exc_info=True
            )
            raise

    @classmethod
    def from_mcp_tool(
        cls, mcp_tool: MCPTool, server: MCPServerProtocol, ignore_errors: bool = False
    ) -> Self:
        """
        Creates a Tool instance from an MCP Tool.

        Args:
            mcp_tool: An MCP Tool object with name, description, and inputSchema.
            server: The MCP server protocol instance.
            ignore_errors: Whether to ignore errors during execution.

        Returns:
            Tool: A new Tool instance.
        """
        _logger.debug(f"Creating Tool from MCP tool: {mcp_tool.name}")

        from mcp.types import (
            BlobResourceContents,
            CallToolResult,
            EmbeddedResource,
            ImageContent,
            TextContent,
            TextResourceContents,
        )

        try:
            tool = cls(
                name=mcp_tool.name,
                description=mcp_tool.description,
                parameters=mcp_tool.inputSchema,
                ignore_errors=ignore_errors,
            )
            tool._server = server

            async def _callable_ref(**kwargs: Any) -> Any:
                _logger.debug(f"Calling MCP tool '{mcp_tool.name}' with server")
                try:
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
                                                data=base64.b64decode(
                                                    content.resource.blob
                                                ),
                                                mime_type="application/octet-stream",
                                            )
                                        )

                    _logger.debug(
                        f"MCP tool '{mcp_tool.name}' returned {len(contents)} content items"
                    )
                    return contents

                except Exception as e:
                    _logger.error(
                        f"Error calling MCP tool '{mcp_tool.name}': {str(e)}",
                        exc_info=True,
                    )
                    raise

            # Use type: ignore to bypass the strict type checking for MCP tools
            # since they have dynamic signatures
            tool._callable_ref = _callable_ref  # type: ignore[assignment]
            _logger.info(f"Successfully created Tool from MCP tool: {mcp_tool.name}")
            return tool

        except Exception as e:
            _logger.error(
                f"Error creating Tool from MCP tool '{mcp_tool.name}': {str(e)}",
                exc_info=True,
            )
            raise

    @classmethod
    def from_callable(
        cls,
        _callable: Callable[CallableP, CallableT]
        | Callable[CallableP, Awaitable[CallableT]],
        /,
        *,
        name: str | None = None,
        description: str | None = None,
        before_call: (
            Callable[CallableP, CallableT | None]
            | Callable[CallableP, Awaitable[CallableT | None]]
            | None
        ) = None,
        after_call: (
            Callable[
                ..., CallableT
            ]  # (result: CallableT, *args, **kwargs) -> CallableT
            | Callable[..., Awaitable[CallableT]]
            | None
        ) = None,
        ignore_errors: bool = False,
    ) -> "Tool[CallableP, CallableT]":
        """
        Creates a Tool instance from a callable function with full type safety.

        This class method analyzes a function's signature and creates a Tool instance
        that preserves both the parameter signature and return type.

        Type Parameters:
            CallableP: ParamSpec for the callable's parameters
            CallableT: Return type of the callable

        Args:
            _callable: A callable function to wrap as a Tool.
            name: Optional custom name for the tool.
            description: Optional custom description for the tool.
            before_call: Optional callback executed before the main function.
                        Receives same params and can optionally return result to short-circuit.
            after_call: Optional callback executed after the main function.
                       Receives (result: CallableT, *args, **kwargs) and can modify the result.
            ignore_errors: Whether to ignore errors during execution.

        Returns:
            Tool[CallableP, CallableT]: A new Tool instance with preserved type signatures.

        Example:
            ```python
            def multiply(a: int, b: int) -> int:
                \"\"\"Multiply two numbers\"\"\"
                return a * b

            def log_before_multiply(a: int, b: int) -> None:
                print(f"About to multiply {a} and {b}")
                return None  # Continue to main function

            def log_after_multiply(result: int, a: int, b: int) -> int:
                print(f"Result: {result}")
                return result

            # Full type safety preserved
            multiply_tool = Tool.from_callable(
                multiply,
                before_call=log_before_multiply,
                after_call=log_after_multiply
            )

            # Type-safe usage - parameters are properly typed!
            result = multiply_tool.call(a=5, b=3)  # Returns int
            ```
        """
        _name: str = name or getattr(_callable, "__name__", "anonymous_function")
        _logger.debug(f"Creating Tool from callable function: {_name}")

        try:
            _description = (
                description or _callable.__doc__ or "No description available"
            )

            # Extract parameter information from the function
            parameters: dict[str, object] = {}
            signature = inspect.signature(_callable)
            _logger.debug(
                f"Analyzing {len(signature.parameters)} parameters for function '{_name}'"
            )

            for param_name, param in signature.parameters.items():
                # Skip self/cls parameters for methods
                if (
                    param_name in ("self", "cls")
                    and param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
                ):
                    _logger.debug(f"Skipping {param_name} parameter (self/cls)")
                    continue

                param_info: dict[str, object] = {"type": "object"}

                # Add type information if available
                if param.annotation != inspect.Parameter.empty:
                    param_type = (
                        str(param.annotation).replace("<class '", "").replace("'>", "")
                    )
                    param_info["type"] = param_type
                    _logger.debug(
                        f"Parameter '{param_name}' has type annotation: {param_type}"
                    )

                # Add default value if available
                if param.default != inspect.Parameter.empty:
                    param_info["default"] = param.default
                    _logger.debug(
                        f"Parameter '{param_name}' has default value: {param.default}"
                    )

                # Determine if parameter is required
                if param.default == inspect.Parameter.empty and param.kind in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                ):
                    param_info["required"] = True
                    _logger.debug(f"Parameter '{param_name}' is required")

                parameters[param_name] = param_info

            # Create instance with type parameter matching the callable's return type
            instance: Tool[CallableP, CallableT] = cls(  # type: ignore[assignment]
                name=_name,
                description=_description,
                parameters=parameters,
                ignore_errors=ignore_errors,
            )

            # Set private attributes after instance creation
            instance._callable_ref = _callable
            instance._before_call = before_call
            instance._after_call = after_call

            _logger.info(
                f"Successfully created Tool from callable: {_name} with {len(parameters)} parameters"
            )
            return instance

        except Exception as e:
            _logger.error(
                f"Error creating Tool from callable '{_name}': {str(e)}", exc_info=True
            )
            raise

    def set_callable_ref(
        self, ref: Callable[P, T_Output] | Callable[P, Awaitable[T_Output]] | None
    ) -> None:
        """Set the callable reference for this tool."""
        self._callable_ref = ref

    def __str__(self) -> str:
        return self.text


# Example usage and type checking
if __name__ == "__main__":
    # Example 1: Simple function
    def add_numbers(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b

    # Tool preserves parameter and return type
    add_tool = Tool.from_callable(add_numbers)
    result = add_tool.call(
        a=5, b=3
    )  # Parameters are fully typed: (a: int, b: int) -> int

    # Example 2: With callbacks
    def log_before(a: int, b: int) -> int | None:
        print(f"About to add {a} and {b}")
        return None  # Continue to main function

    def modify_result(result: int, *args: Any, **kwargs: Any) -> int:
        print(f"Original result: {result}")
        return result * 2  # Double the result

    enhanced_tool = Tool.from_callable(
        add_numbers, before_call=log_before, after_call=modify_result
    )

    enhanced_result = enhanced_tool.call(a=5, b=3)  # result is 16 (8*2)
