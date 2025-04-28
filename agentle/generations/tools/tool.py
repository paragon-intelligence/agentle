from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from rsb.models.base_model import BaseModel
from rsb.models.config_dict import ConfigDict
from rsb.models.field import Field
from rsb.models.private_attr import PrivateAttr

if TYPE_CHECKING:
    from mcp.types import Tool as MCPTool


class Tool[T_Output = Any](BaseModel):
    type: Literal["tool"] = Field(default="tool")
    name: str
    description: str | None = Field(default=None)
    parameters: dict[str, object]
    _callable_ref: Callable[..., T_Output] | None = PrivateAttr(default=None)
    needs_human_confirmation: bool = Field(default=False)

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    @property
    def text(self) -> str:
        return f"Tool: {self.name}\nDescription: {self.description}\nParameters: {self.parameters}"

    def call(self, **kwargs: object) -> T_Output:
        if self._callable_ref is None:
            raise ValueError(
                'Tool is not callable because the "_callable_ref" instance variable is not set'
            )

        return self._callable_ref(**kwargs)

    @classmethod
    def from_mcp_tool(cls, mcp_tool: MCPTool) -> Tool[T_Output]:
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
