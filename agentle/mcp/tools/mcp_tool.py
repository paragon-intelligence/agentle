from __future__ import annotations

import inspect
from typing import Callable
from agentle.mcp.tools.input_schema import InputSchema
from rsb.models.base_model import BaseModel


class MCPTool(BaseModel):
    """Definition for a tool the client can call."""

    name: str
    """The name of the tool."""
    description: str | None = None
    """A human-readable description of the tool."""
    inputSchema: InputSchema
    """A JSON Schema object defining the expected parameters for the tool."""

    @classmethod
    def from_callable(cls, _callable: Callable[..., object], /) -> MCPTool:
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
