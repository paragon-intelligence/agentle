from __future__ import annotations

from collections.abc import Callable
import inspect
from typing import Literal
from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class ToolDeclaration(BaseModel):
    type: Literal["tool_declaration"] = Field(default="tool_declaration")
    name: str
    description: str | None = None
    parameters: dict[str, object]
    callable_ref: Callable[..., object] | None = None
    needs_human_confirmation: bool = False

    @property
    def text(self) -> str:
        return f"Tool: {self.name}\nDescription: {self.description}\nParameters: {self.parameters}"

    @classmethod
    def from_callable(
        cls,
        _callable: Callable[..., object],
        /,
    ) -> ToolDeclaration:
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

        return cls(
            name=name,
            description=description,
            callable_ref=_callable,
            parameters=parameters,
        )
