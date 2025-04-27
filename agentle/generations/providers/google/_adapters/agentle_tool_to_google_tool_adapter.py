from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from rsb.adapters.adapter import Adapter

from agentle.generations.tools.tool import Tool

if TYPE_CHECKING:
    from google.genai import types


class AgentleToolToGoogleToolAdapter(Adapter[Tool[Any], "types.Tool"]):
    def adapt(self, agentle_tool: Tool[Any]) -> types.Tool:
        from google.genai import types

        # Mapeamento de tipos de string para google.genai.types.Type
        type_mapping = {
            "str": types.Type.STRING,
            "string": types.Type.STRING,
            "int": types.Type.INTEGER,
            "integer": types.Type.INTEGER,
            "float": types.Type.NUMBER,
            "number": types.Type.NUMBER,
            "bool": types.Type.BOOLEAN,
            "boolean": types.Type.BOOLEAN,
            "list": types.Type.ARRAY,
            "array": types.Type.ARRAY,
            "dict": types.Type.OBJECT,
            "object": types.Type.OBJECT,
        }

        properties: dict[str, types.Schema] = {}
        required: list[str] = []

        for param_name, param_info_obj in agentle_tool.parameters.items():
            # Cast para dict para ajudar o linter
            param_info = cast(dict[str, Any], param_info_obj)
            param_schema_info: dict[str, Any] = {}

            # Mapear o tipo
            param_type_str = param_info.get("type", "object")
            google_type = type_mapping.get(
                str(param_type_str).lower(), types.Type.OBJECT
            )  # Default to OBJECT if type unknown
            param_schema_info["type"] = google_type

            # TODO: Adicionar description se disponível em param_info futuramente
            # param_schema_info["description"] = param_info.get("description")

            # Adicionar valor padrão se disponível
            if "default" in param_info:
                param_schema_info["default"] = param_info["default"]

            # Adicionar items para arrays (listas) - Simplificado, assume items são strings por agora
            if google_type == types.Type.ARRAY:
                # Assume array de strings como padrão simplificado
                # Idealmente, o agentle.Tool.parameters precisaria especificar o tipo dos itens
                param_schema_info["items"] = types.Schema(type=types.Type.STRING)

            properties[param_name] = types.Schema(**param_schema_info)

            if param_info.get("required", False):
                required.append(param_name)

        # Criar o schema principal para os parâmetros da função
        parameters_schema = types.Schema(type=types.Type.OBJECT, properties=properties)
        # Só adicionar 'required' se a lista não estiver vazia
        if required:
            parameters_schema.required = required

        # Criar a declaração da função
        function_declaration = types.FunctionDeclaration(
            name=agentle_tool.name,
            description=agentle_tool.description
            or "",  # Usar string vazia se a descrição for None
            parameters=parameters_schema,
        )

        # Criar e retornar a ferramenta do Google
        return types.Tool(function_declarations=[function_declaration])
