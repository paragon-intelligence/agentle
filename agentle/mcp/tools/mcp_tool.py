"""
MCP Tool definition module.

This module provides the MCPTool class, which represents a callable tool in the
Model Control Protocol (MCP) system. MCPTools encapsulate functions or methods
with their metadata and input schemas, making them available for discovery and
execution by MCP clients.
"""

from __future__ import annotations

import inspect
from typing import Callable
from agentle.mcp.tools.input_schema import InputSchema
from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class MCPTool(BaseModel):
    """
    Definition for a tool that clients can call in the MCP system.

    An MCPTool represents a callable function or method with its associated metadata
    and input schema. It provides information about the tool's name, description,
    and expected parameters that clients can use for discovery and execution.

    Tools can be created directly or from existing callables using the `from_callable`
    class method, which automatically extracts parameter information.

    Attributes:
        name (str): The name of the tool
        description (str | None): A human-readable description of the tool
        inputSchema (InputSchema): A schema defining the expected parameters
    """

    name: str = Field(..., description="The name of the tool.")
    """The name of the tool."""
    description: str | None = Field(
        default=None, description="A human-readable description of the tool."
    )
    """A human-readable description of the tool."""
    inputSchema: InputSchema = Field(
        ...,
        description="A JSON Schema object defining the expected parameters for the tool.",
    )
    """A JSON Schema object defining the expected parameters for the tool."""

    @classmethod
    def from_callable(cls, _callable: Callable[..., object], /) -> MCPTool:
        """
        Create an MCPTool from a callable function or method.

        This method inspects the provided callable and extracts metadata to create
        an MCPTool instance. It automatically determines:
        - Tool name from the function name
        - Description from the function docstring
        - Input schema from parameter annotations, defaults, and metadata

        The method handles special cases such as self/cls parameters for methods
        and extracts type information and descriptions when available.

        Args:
            _callable (Callable[..., object]): The function or method to convert
                to an MCPTool

        Returns:
            MCPTool: A new MCPTool instance representing the callable

        Example:
            ```python
            def add(a: int, b: int) -> int:
                '''Add two numbers together.'''
                return a + b

            tool = MCPTool.from_callable(add)
            # Creates a tool with name="add", description="Add two numbers together."
            # and appropriate input schema for the parameters
            ```
        """
        name = getattr(_callable, "__name__", "anonymous_function")
        description = _callable.__doc__ or None

        # Extrair informações dos parâmetros da função
        properties: dict[str, object] = {}
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

            # Adicionar descrição se disponível através de metadados
            if hasattr(_callable, "__annotations_metadata__") and param_name in getattr(
                _callable, "__annotations_metadata__", {}
            ):
                metadata = getattr(_callable, "__annotations_metadata__")[param_name]
                if "description" in metadata:
                    param_info["description"] = metadata["description"]

            properties[param_name] = param_info

        # Criar o schema de entrada
        input_schema = InputSchema(properties=properties)

        return cls(
            name=name,
            description=description,
            inputSchema=input_schema,
        )
