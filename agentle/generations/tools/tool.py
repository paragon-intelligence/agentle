"""
Enhanced Tool module with robust serialization support using dill.

This version adds comprehensive serialization capabilities that allow Tools to be
fully serialized and deserialized while preserving their callable references.
Assumes dill is always available.
"""

from __future__ import annotations

import base64
import inspect
import logging
import sys
from collections.abc import Awaitable, Callable, MutableSequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, Generic, Literal, ParamSpec, TypedDict, TypeVar
import warnings

import dill
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

warnings.filterwarnings("ignore", category=dill.PicklingWarning)

# Type variables for the from_callable method
CallableP = ParamSpec("CallableP")
CallableT = TypeVar("CallableT")

P = ParamSpec(
    "P",
    default=...,  # Can't use Python's 3.13 new Generic syntax. Cannot pickle them.
)
T_Output = TypeVar("T_Output", default=Any)
ApprovalPolicy = Literal[
    "never",
    "always",
    "required",
    "sensitive",
    "human",
    "human_in_the_loop",
    "hitl",
]
RiskLevel = Literal["low", "medium", "high", "critical"]


class _SerializationError(Exception):
    """Raised when tool serialization/deserialization fails."""

    pass


class _SerializationSizes(TypedDict):
    main: int
    before_call: int
    after_call: int


class _SerializationInfo(TypedDict):
    tool_name: str
    is_serializable: bool
    is_mcp_tool: bool
    has_callable_ref: bool
    callables_reconstructed: bool
    serialization_metadata: dict[str, Any]
    serialized_sizes: _SerializationSizes
    total_serialized_size: int


class Tool(BaseModel, Generic[P, T_Output]):
    """
    A callable tool with robust serialization support using dill.

    This enhanced version of Tool includes comprehensive serialization capabilities
    that preserve callable references, making it suitable for evaluation frameworks
    and distributed systems where tools need to be serialized and reconstructed.

    The serialization system:
    - Uses dill to serialize callable functions, preserving closures and complex objects
    - Stores serialized code in base64-encoded fields for JSON compatibility
    - Provides automatic reconstruction of callables after deserialization
    - Handles edge cases and provides graceful error recovery
    - Validates reconstructed callables to ensure they work correctly

    Type Parameters:
        P: ParamSpec for the callable's parameters
        T_Output: Return type of the callable function

    Attributes:
        type: Literal field that identifies this as a tool, always set to "tool".
        name: Human-readable name of the tool.
        description: Human-readable description of what the tool does.
        parameters: Dictionary of parameter specifications for the tool.
        code: Base64-encoded serialized main callable function.
        before_call_code: Base64-encoded serialized before_call callback.
        after_call_code: Base64-encoded serialized after_call callback.
        serialization_metadata: Metadata about the serialization process.
        _callable_ref: Private attribute storing the callable function.
        _before_call: Optional callback executed before the main function.
        _after_call: Optional callback executed after the main function.
        _server: MCP server reference for MCP tools.
    """

    type: Literal["tool"] = Field(
        default="tool",
        description="Discriminator field identifying this as a tool object.",
    )

    name: str = Field(
        description="Human-readable name of the tool, used for identification and display.",
    )

    description: str | None = Field(
        default=None,
        description="Human-readable description of what the tool does and how to use it.",
    )

    parameters: dict[str, Any] = Field(
        description="Dictionary of parameter specifications for the tool, including types, descriptions, and constraints.",
    )

    ignore_errors: bool = Field(
        default=False,
        description="If True, errors in the tool execution will be ignored and the agent will continue running.",
    )

    approval_required: bool = Field(
        default=False,
        description="Whether this tool must be approved by a human before dispatch.",
    )

    approval_policy: ApprovalPolicy | None = Field(
        default=None,
        description="Approval policy name for human approval workflows.",
    )

    risk_level: RiskLevel | None = Field(
        default=None,
        description="Human-readable risk level for approval UIs and policy engines.",
    )

    approval_timeout_seconds: int | None = Field(
        default=None,
        description="How long an approval request may remain pending before expiring.",
    )

    approver_scope: dict[str, Any] | None = Field(
        default=None,
        description="Structured scope describing who may approve this tool execution.",
    )

    display_template: str | None = Field(
        default=None,
        description="Optional template used by applications to display approval requests.",
    )

    redaction_schema: dict[str, Any] | None = Field(
        default=None,
        description="Optional schema describing how applications should redact tool arguments.",
    )

    approval_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Application-specific metadata for human approval workflows.",
    )

    # Serialization fields - store base64-encoded dill-serialized callables
    code: str | None = Field(
        default=None,
        description="Base64-encoded serialized main callable function for reconstruction after deserialization.",
    )

    before_call_code: str | None = Field(
        default=None,
        description="Base64-encoded serialized before_call callback function.",
    )

    after_call_code: str | None = Field(
        default=None,
        description="Base64-encoded serialized after_call callback function.",
    )

    serialization_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Metadata about the serialization process including dill version, python version, etc.",
    )

    # Private attributes for runtime callables
    _callable_ref: Callable[P, T_Output] | Callable[P, Awaitable[T_Output]] | None = (
        PrivateAttr(default=None)
    )

    _before_call: (
        Callable[P, T_Output | None] | Callable[P, Awaitable[T_Output | None]] | None
    ) = PrivateAttr(default=None)

    _after_call: Callable[..., T_Output] | Callable[..., Awaitable[T_Output]] | None = (
        PrivateAttr(default=None)
    )

    _server: MCPServerProtocol | None = PrivateAttr(default=None)

    # Flag to track if callables have been reconstructed
    _callables_reconstructed: bool = PrivateAttr(default=False)

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=False)

    def model_post_init(self, __context: Any) -> None:
        """Post-initialization hook to reconstruct callables if needed."""
        super().model_post_init(__context)

        # If we have serialized code but no callable ref, try to reconstruct
        if (
            self.code is not None
            and self._callable_ref is None
            and not self._callables_reconstructed
        ):
            try:
                self._reconstruct_callables()
            except Exception as e:
                _logger.warning(
                    f"Failed to reconstruct callables during init for tool '{self.name}': {e}"
                )

    def is_mcp_tool(self) -> bool:
        """Check if this is an MCP tool."""
        return self._server is not None

    def is_serializable(self) -> bool:
        """Check if this tool has serializable callables."""
        return self.code is not None

    @property
    def callable_ref(
        self,
    ) -> Callable[P, T_Output] | Callable[P, Awaitable[T_Output]] | None:
        """Get the reconstructed callable reference."""
        return self._callable_ref

    @property
    def text(self) -> str:
        """
        Generates a human-readable text representation of the tool.

        Returns:
            str: A formatted string containing the tool name, description, and parameters.
        """
        return f"Tool: {self.name}\nDescription: {self.description}\nParameters: {self.parameters}"

    def ensure_callable_available(self) -> None:
        """
        Ensure the callable reference is available, reconstructing from code if necessary.

        Raises:
            _SerializationError: If callable cannot be made available
            ValueError: If tool is not callable and cannot be reconstructed
        """
        if self._callable_ref is not None:
            return  # Already available

        if self.code is not None and not self._callables_reconstructed:
            try:
                self._reconstruct_callables()
                if self._callable_ref is not None:
                    return
            except _SerializationError:
                raise
            except Exception as e:
                raise _SerializationError(
                    f"Failed to reconstruct callable for tool '{self.name}': {e}"
                ) from e

        # Final check
        if self._callable_ref is None:
            if self.is_mcp_tool():
                # MCP tools don't need pre-existing callables
                return
            else:
                raise ValueError(
                    f"Tool '{self.name}' has no available callable and cannot be reconstructed from serialized code"
                )

    def validate_callable(self) -> bool:
        """
        Validate that the callable reference works correctly.

        Returns:
            bool: True if callable is valid and working, False otherwise
        """
        try:
            self.ensure_callable_available()

            if self._callable_ref is None:
                if self.is_mcp_tool():
                    return True  # MCP tools are handled differently
                return False

            # Basic signature validation
            if hasattr(self._callable_ref, "__call__"):
                try:
                    signature = inspect.signature(self._callable_ref)
                    _logger.debug(
                        f"Callable signature validated for '{self.name}': {signature}"
                    )
                    return True
                except (ValueError, TypeError) as e:
                    _logger.warning(
                        f"Signature validation failed for '{self.name}': {e}"
                    )
                    return False

            return False

        except Exception as e:
            _logger.error(f"Callable validation failed for tool '{self.name}': {e}")
            return False

    def call(self, *args: P.args, **kwargs: P.kwargs) -> T_Output:
        """
        Executes the underlying function with the provided arguments.
        Automatically reconstructs callables if needed.

        Args:
            *args: Positional arguments matching the ParamSpec P of the underlying function.
            **kwargs: Keyword arguments matching the ParamSpec P of the underlying function.

        Returns:
            T_Output: The result of calling the underlying function.

        Raises:
            _SerializationError: If callable cannot be made available.
            ValueError: If the Tool does not have a callable reference.
        """
        # Ensure callable is available before proceeding
        self.ensure_callable_available()

        ret = run_sync(self.call_async, timeout=None, *args, **kwargs)
        return ret

    def _convert_parameter_types(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Convert parameter values to their expected types based on function signature.

        This is particularly important for complex types like BaseModel, TypedDict,
        dataclasses, etc., where the AI passes a dict but the function expects
        an instance of the type.

        Args:
            kwargs: Keyword arguments to convert.

        Returns:
            Converted keyword arguments.
        """
        if self._callable_ref is None:
            return kwargs

        try:
            sig = inspect.signature(self._callable_ref)
            converted_kwargs = {}

            for param_name, param_value in kwargs.items():
                if param_name not in sig.parameters:
                    # Keep unknown parameters as-is
                    converted_kwargs[param_name] = param_value
                    continue

                param = sig.parameters[param_name]

                # Skip if no annotation
                if param.annotation == inspect.Parameter.empty:
                    converted_kwargs[param_name] = param_value
                    continue

                param_type = param.annotation

                # Check if we need to convert from dict to a complex type
                if isinstance(param_value, dict) and inspect.isclass(param_type):
                    # Check if it's a BaseModel (Pydantic)
                    if hasattr(param_type, "model_validate"):
                        # Pydantic v2
                        try:
                            converted_kwargs[param_name] = param_type.model_validate(
                                param_value
                            )
                            _logger.debug(
                                f"Converted parameter '{param_name}' from dict to {param_type.__name__} (Pydantic v2)"
                            )
                            continue
                        except Exception as e:
                            _logger.warning(
                                f"Failed to convert '{param_name}' using Pydantic v2: {e}"
                            )

                    # Check if it's a Pydantic v1 model
                    if hasattr(param_type, "parse_obj"):
                        try:
                            converted_kwargs[param_name] = param_type.parse_obj(
                                param_value
                            )
                            _logger.debug(
                                f"Converted parameter '{param_name}' from dict to {param_type.__name__} (Pydantic v1)"
                            )
                            continue
                        except Exception as e:
                            _logger.warning(
                                f"Failed to convert '{param_name}' using Pydantic v1: {e}"
                            )

                    # Check if it's a dataclass
                    if hasattr(param_type, "__dataclass_fields__"):
                        try:
                            converted_kwargs[param_name] = param_type(**param_value)
                            _logger.debug(
                                f"Converted parameter '{param_name}' from dict to {param_type.__name__} (dataclass)"
                            )
                            continue
                        except Exception as e:
                            _logger.warning(
                                f"Failed to convert '{param_name}' to dataclass: {e}"
                            )

                    # Check if it's a TypedDict (these stay as dicts, no conversion needed)
                    if hasattr(param_type, "__annotations__") and hasattr(
                        param_type, "__required_keys__"
                    ):
                        # TypedDict is just a type hint, the value is already a dict
                        converted_kwargs[param_name] = param_value
                        _logger.debug(
                            f"Parameter '{param_name}' is TypedDict, keeping as dict"
                        )
                        continue

                # No conversion needed or possible, keep original value
                converted_kwargs[param_name] = param_value

            return converted_kwargs

        except Exception as e:
            _logger.warning(
                f"Failed to convert parameter types for tool '{self.name}': {e}. "
                + "Using original parameters."
            )
            return kwargs

    async def call_async(self, *args: P.args, **kwargs: P.kwargs) -> T_Output:
        """
        Executes the underlying function asynchronously with the provided arguments.
        Automatically reconstructs callables if needed and converts parameter types.

        Args:
            *args: Positional arguments matching the ParamSpec P of the underlying function.
            **kwargs: Keyword arguments matching the ParamSpec P of the underlying function.

        Returns:
            T_Output: The result of calling the underlying function.

        Raises:
            _SerializationError: If callable cannot be made available.
            ValueError: If the Tool does not have a callable reference.
        """
        _logger.debug(
            f"Calling tool '{self.name}' with arguments: args={args}, kwargs={kwargs}"
        )

        # Ensure callable is available before proceeding
        self.ensure_callable_available()

        if self._callable_ref is None:
            _logger.error(f"Tool '{self.name}' is not callable - missing _callable_ref")
            raise ValueError(
                f'Tool "{self.name}" is not callable because the "_callable_ref" instance variable is not set'
            )

        # Convert parameter types (e.g., dict to BaseModel)
        converted_kwargs = self._convert_parameter_types(dict(kwargs))

        try:
            # Execute before_call callback - can short-circuit execution
            if self._before_call is not None:
                _logger.debug(f"Executing before_call callback for tool '{self.name}'")
                if inspect.iscoroutinefunction(self._before_call):
                    before_result = await self._before_call(*args, **converted_kwargs)
                else:
                    before_result = self._before_call(*args, **converted_kwargs)

                # If before_call returns a result, use it and skip main function
                if before_result is not None:
                    _logger.debug("before_call returned result, skipping main function")
                    return before_result  # type: ignore[return-value]

            # Execute the main function
            _logger.debug(f"Executing main function for tool '{self.name}'")
            if inspect.iscoroutinefunction(self._callable_ref):
                try:
                    async_ret: T_Output = await self._callable_ref(
                        *args, **converted_kwargs
                    )
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
                    sync_ret: T_Output = self._callable_ref(*args, **converted_kwargs)  # type: ignore[misc]
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
                    modified_result = await self._after_call(
                        ret, *args, **converted_kwargs
                    )
                else:
                    modified_result = self._after_call(ret, *args, **converted_kwargs)

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
    ) -> Tool[..., Any]:
        """
        Creates a Tool instance from an MCP Tool.

        Note: MCP tools cannot be serialized due to server dependency.

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

            # Set callable reference for MCP tool
            tool._callable_ref = _callable_ref  # type: ignore[assignment]

            _logger.info(f"Successfully created Tool from MCP tool: {mcp_tool.name}")
            _logger.warning(
                f"MCP tool '{mcp_tool.name}' cannot be serialized due to server dependency"
            )
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
        auto_serialize: bool = True,
        approval_required: bool | None = None,
        approval_policy: ApprovalPolicy | None = None,
        risk_level: RiskLevel | None = None,
        approval_timeout_seconds: int | None = None,
        approver_scope: dict[str, Any] | None = None,
        display_template: str | None = None,
        redaction_schema: dict[str, Any] | None = None,
        approval_metadata: dict[str, Any] | None = None,
    ) -> Tool[CallableP, CallableT]:
        """
        Creates a Tool instance from a callable function with full type safety and serialization.

        This class method analyzes a function's signature and creates a Tool instance
        that preserves both the parameter signature and return type. If auto_serialize
        is True, it will automatically serialize the callable for later reconstruction.

        Type Parameters:
            CallableP: ParamSpec for the callable's parameters
            CallableT: Return type of the callable

        Args:
            _callable: A callable function to wrap as a Tool.
            name: Optional custom name for the tool.
            description: Optional custom description for the tool.
            before_call: Optional callback executed before the main function.
            after_call: Optional callback executed after the main function.
            ignore_errors: Whether to ignore errors during execution.
            auto_serialize: Whether to automatically serialize callables (default: True).

        Returns:
            Tool[CallableP, CallableT]: A new Tool instance with preserved type signatures.

        Raises:
            _SerializationError: If auto_serialize is True and serialization fails.

        Example:
            ```python
            def multiply(a: int, b: int) -> int:
                \"\"\"Multiply two numbers\"\"\"
                return a * b

            # Create tool with automatic serialization
            multiply_tool = Tool.from_callable(multiply)

            # The tool can now be serialized/deserialized
            import dill
            serialized = dill.dumps(multiply_tool)
            reconstructed = dill.loads(serialized)

            # Callable is automatically reconstructed
            result = reconstructed.call(a=5, b=3)  # Works!
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
                approval_required=bool(
                    approval_required
                    if approval_required is not None
                    else getattr(_callable, "approval_required", False)
                ),
                approval_policy=approval_policy
                if approval_policy is not None
                else getattr(_callable, "approval_policy", None),
                risk_level=risk_level
                if risk_level is not None
                else getattr(_callable, "risk_level", None),
                approval_timeout_seconds=approval_timeout_seconds
                if approval_timeout_seconds is not None
                else getattr(_callable, "approval_timeout_seconds", None),
                approver_scope=approver_scope
                if approver_scope is not None
                else getattr(_callable, "approver_scope", None),
                display_template=display_template
                if display_template is not None
                else getattr(_callable, "display_template", None),
                redaction_schema=redaction_schema
                if redaction_schema is not None
                else getattr(_callable, "redaction_schema", None),
                approval_metadata=approval_metadata
                if approval_metadata is not None
                else dict(getattr(_callable, "approval_metadata", {}) or {}),
            )

            # Set private attributes after instance creation
            instance._callable_ref = _callable
            instance._before_call = before_call
            instance._after_call = after_call

            # Automatically serialize callables if requested
            if auto_serialize:
                try:
                    instance._serialize_all_callables()
                    _logger.debug(f"Auto-serialized callables for tool '{_name}'")
                except _SerializationError:
                    raise  # Re-raise _SerializationErrors
                except Exception as e:
                    raise _SerializationError(
                        f"Auto-serialization failed for tool '{_name}': {e}"
                    ) from e

            _logger.info(
                f"Successfully created Tool from callable: {_name} with {len(parameters)} parameters"
            )
            return instance

        except _SerializationError:
            raise  # Re-raise _SerializationErrors
        except Exception as e:
            _logger.error(
                f"Error creating Tool from callable '{_name}': {str(e)}", exc_info=True
            )
            raise

    def set_callable_ref(
        self, ref: Callable[P, T_Output] | Callable[P, Awaitable[T_Output]] | None
    ) -> None:
        """
        Set the callable reference for this tool.

        Args:
            ref: The callable reference to set
        """
        self._callable_ref = ref
        # Reset reconstruction flag since we have a new callable
        self._callables_reconstructed = False

    def force_serialize(self) -> None:
        """
        Force serialization of all callables, even if they were already serialized.
        Useful when callables have been modified after creation.

        Raises:
            _SerializationError: If serialization fails.
        """
        # Reset reconstruction flag to force fresh serialization
        self._callables_reconstructed = False
        self._serialize_all_callables()
        _logger.info(f"Forced serialization completed for tool '{self.name}'")

    def clear_serialization(self) -> None:
        """
        Clear all serialized code fields. Useful for reducing size when callables
        are no longer needed for serialization.
        """
        self.code = None
        self.before_call_code = None
        self.after_call_code = None
        self.serialization_metadata = {}
        self._callables_reconstructed = False
        _logger.debug(f"Cleared serialization data for tool '{self.name}'")

    def get_serialization_info(self) -> _SerializationInfo:
        """
        Get comprehensive information about the serialization state of this tool.

        Returns:
            dict: Serialization state information
        """
        return {
            "tool_name": self.name,
            "is_serializable": self.is_serializable(),
            "is_mcp_tool": self.is_mcp_tool(),
            "has_callable_ref": self._callable_ref is not None,
            "callables_reconstructed": self._callables_reconstructed,
            "serialization_metadata": self.serialization_metadata,
            "serialized_sizes": {
                "main": len(self.code) if self.code else 0,
                "before_call": len(self.before_call_code)
                if self.before_call_code
                else 0,
                "after_call": len(self.after_call_code) if self.after_call_code else 0,
            },
            "total_serialized_size": sum(
                [
                    len(self.code) if self.code else 0,
                    len(self.before_call_code) if self.before_call_code else 0,
                    len(self.after_call_code) if self.after_call_code else 0,
                ]
            ),
        }

    def __str__(self) -> str:
        return self.text

    def _serialize_callable(
        self, callable_obj: Any, name: str = "anonymous"
    ) -> str | None:
        """
        Serialize a callable using dill and encode as base64.

        Args:
            callable_obj: The callable to serialize
            name: Name for logging purposes

        Returns:
            Base64-encoded serialized callable, or None if serialization fails

        Raises:
            _SerializationError: If serialization fails critically
        """
        if callable_obj is None:
            return None

        try:
            # Use dill's highest protocol for efficiency
            serialized_bytes = dill.dumps(callable_obj, protocol=dill.HIGHEST_PROTOCOL)
            # Encode as base64 for JSON compatibility
            encoded = base64.b64encode(serialized_bytes).decode("utf-8")

            _logger.debug(
                f"Successfully serialized callable '{name}': {len(encoded)} chars"
            )
            return encoded

        except Exception as e:
            _logger.error(f"Failed to serialize callable '{name}': {e}")
            raise _SerializationError(
                f"Failed to serialize callable '{name}': {e}"
            ) from e

    def _deserialize_callable(
        self, encoded_code: str | None, name: str = "anonymous"
    ) -> Any:
        """
        Deserialize a callable from base64-encoded dill data.

        Args:
            encoded_code: Base64-encoded serialized callable
            name: Name for logging purposes

        Returns:
            The reconstructed callable, or None if deserialization fails

        Raises:
            _SerializationError: If deserialization fails critically
        """
        if encoded_code is None:
            return None

        try:
            # Decode from base64
            serialized_bytes = base64.b64decode(encoded_code.encode("utf-8"))
            # Deserialize using dill
            callable_obj = dill.loads(serialized_bytes)

            _logger.debug(f"Successfully deserialized callable '{name}'")
            return callable_obj

        except Exception as e:
            _logger.error(f"Failed to deserialize callable '{name}': {e}")
            raise _SerializationError(
                f"Failed to deserialize callable '{name}': {e}"
            ) from e

    def _serialize_all_callables(self) -> None:
        """
        Serialize all callable references and store them in the code fields.
        Also stores metadata about the serialization process.

        Raises:
            _SerializationError: If serialization fails
        """
        try:
            # Serialize main callable
            if self._callable_ref is not None:
                self.code = self._serialize_callable(
                    self._callable_ref, f"{self.name}_main"
                )

            # Serialize callbacks
            if self._before_call is not None:
                self.before_call_code = self._serialize_callable(
                    self._before_call, f"{self.name}_before"
                )

            if self._after_call is not None:
                self.after_call_code = self._serialize_callable(
                    self._after_call, f"{self.name}_after"
                )

            # Store comprehensive serialization metadata
            self.serialization_metadata = {
                "dill_version": dill.__version__,
                "python_version": sys.version,
                "python_version_info": {
                    "major": sys.version_info.major,
                    "minor": sys.version_info.minor,
                    "micro": sys.version_info.micro,
                },
                "serialized_at": datetime.now().isoformat(),
                "tool_name": self.name,
                "has_main_callable": self.code is not None,
                "has_before_callback": self.before_call_code is not None,
                "has_after_callback": self.after_call_code is not None,
                "main_callable_size": len(self.code) if self.code else 0,
                "before_callback_size": len(self.before_call_code)
                if self.before_call_code
                else 0,
                "after_callback_size": len(self.after_call_code)
                if self.after_call_code
                else 0,
            }

            _logger.info(
                f"Successfully serialized all callables for tool '{self.name}'"
            )

        except _SerializationError:
            raise  # Re-raise _SerializationErrors
        except Exception as e:
            _logger.error(f"Unexpected error during callable serialization: {e}")
            raise _SerializationError(
                f"Failed to serialize callables for tool '{self.name}': {e}"
            ) from e

    def _reconstruct_callables(self) -> None:
        """
        Reconstruct callable references from serialized code fields.

        Raises:
            _SerializationError: If reconstruction fails
        """
        if self._callables_reconstructed:
            _logger.debug(
                f"Callables already reconstructed for tool '{self.name}', skipping"
            )
            return

        try:
            # Reconstruct main callable
            if self.code is not None:
                reconstructed = self._deserialize_callable(
                    self.code, f"{self.name}_main"
                )
                if reconstructed is not None:
                    self._callable_ref = reconstructed
                    _logger.debug(
                        f"Successfully reconstructed main callable for tool '{self.name}'"
                    )

            # Reconstruct callbacks
            if self.before_call_code is not None:
                reconstructed = self._deserialize_callable(
                    self.before_call_code, f"{self.name}_before"
                )
                if reconstructed is not None:
                    self._before_call = reconstructed
                    _logger.debug(
                        f"Successfully reconstructed before_call callback for tool '{self.name}'"
                    )

            if self.after_call_code is not None:
                reconstructed = self._deserialize_callable(
                    self.after_call_code, f"{self.name}_after"
                )
                if reconstructed is not None:
                    self._after_call = reconstructed
                    _logger.debug(
                        f"Successfully reconstructed after_call callback for tool '{self.name}'"
                    )

            self._callables_reconstructed = True

            # Update metadata
            if "reconstructed_at" not in self.serialization_metadata:
                self.serialization_metadata["reconstructed_at"] = (
                    datetime.now().isoformat()
                )
                self.serialization_metadata["reconstruction_python_version"] = (
                    sys.version
                )

            _logger.info(f"Successfully reconstructed callables for tool '{self.name}'")

        except _SerializationError:
            raise  # Re-raise _SerializationErrors
        except Exception as e:
            _logger.error(f"Unexpected error during callable reconstruction: {e}")
            raise _SerializationError(
                f"Failed to reconstruct callables for tool '{self.name}': {e}"
            ) from e


# Example usage demonstrating serialization capabilities
if __name__ == "__main__":
    # Example 1: Basic serialization with complex closures
    def create_multiplier_factory(base_multiplier: float):
        """Factory that creates multiplier functions with closures."""

        def multiply_with_base(x: float, y: float) -> float:
            """Multiply two numbers and apply base multiplier."""
            return (x * y) * base_multiplier

        return multiply_with_base

    # Create a function with closure
    double_multiplier = create_multiplier_factory(2.0)

    # Create tool with automatic serialization
    multiply_tool = Tool.from_callable(double_multiplier, name="double_multiply")
    print(f"Original tool callable available: {multiply_tool.callable_ref is not None}")
    print(f"Tool is serializable: {multiply_tool.is_serializable()}")
    print(f"Serialization info: {multiply_tool.get_serialization_info()}")

    # Serialize the entire tool using dill
    serialized_tool = dill.dumps(multiply_tool)
    print(f"Tool serialized successfully: {len(serialized_tool)} bytes")

    # Deserialize and test
    reconstructed_tool = dill.loads(serialized_tool)
    result = reconstructed_tool.call(x=5.0, y=3.0)  # Should be 30.0 (5*3*2)
    print(f"Reconstructed tool result: {result}")
    print(f"Callable validation: {reconstructed_tool.validate_callable()}")

    # Example 2: With callbacks and complex state
    class StatefulProcessor:
        def __init__(self, initial_count: int = 0):
            self.count = initial_count

        def log_before(self, a: int, b: int) -> int | None:
            self.count += 1
            print(f"Call #{self.count}: About to add {a} and {b}")
            return None  # Continue to main function

        def modify_result(self, result: int, *args: Any, **kwargs: Any) -> int:
            print(f"Original result: {result}, doubling it")
            return result * 2

    processor = StatefulProcessor(100)

    def add_numbers(a: int, b: int) -> int:
        """Add two numbers together."""
        return a + b

    enhanced_tool = Tool.from_callable(
        add_numbers,
        before_call=processor.log_before,
        after_call=processor.modify_result,
        name="stateful_adder",
    )

    # Test before serialization
    print("=== Before serialization ===")
    result1 = enhanced_tool.call(a=5, b=3)  # Should print call #101 and return 16
    print(f"Result: {result1}")

    # Test serialization with stateful callbacks
    try:
        serialized_enhanced = dill.dumps(enhanced_tool)
        reconstructed_enhanced = dill.loads(serialized_enhanced)

        print("=== After serialization/deserialization ===")
        result2 = reconstructed_enhanced.call(
            a=7, b=2
        )  # Should continue from preserved state
        print(f"Result: {result2}")

    except _SerializationError as e:
        print(f"Serialization failed: {e}")

    # Example 3: Manual serialization control and validation
    def complex_function(data: dict[str, Any]) -> str:
        """A complex function that processes dictionary data."""
        import json

        processed = {
            k: v * 2 if isinstance(v, (int, float)) else v for k, v in data.items()
        }
        return json.dumps(processed, sort_keys=True)

    manual_tool = Tool.from_callable(complex_function, auto_serialize=False)
    print(f"Manual tool has code: {manual_tool.code is not None}")

    # Serialize manually when needed
    try:
        manual_tool.force_serialize()
        print(f"After manual serialization: {manual_tool.code is not None}")
        print(f"Serialization info: {manual_tool.get_serialization_info()}")

        # Test the reconstructed callable
        test_data = {"a": 5, "b": "hello", "c": 3.14}
        result = manual_tool.call(data=test_data)
        print(f"Manual tool result: {result}")

    except _SerializationError as e:
        print(f"Manual serialization failed: {e}")

    # Validate the callable works correctly
    print(f"Callable validation: {manual_tool.validate_callable()}")
