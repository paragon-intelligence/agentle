"""
Adapter module for converting Agentle Tool objects to Google AI Tool format.

This module provides the AgentleToolToGoogleToolAdapter class, which transforms
Agentle's internal Tool representation into the Tool format expected by Google's
Generative AI APIs. This conversion is necessary when using Agentle tools with
Google's AI models that support function calling capabilities.

The adapter handles the mapping of Agentle tool definitions, including parameters,
types, and descriptions, to Google's schema-based function declaration format.
It includes comprehensive type mapping between Agentle's string-based types and
Google's enumerated Type values.

This adapter is typically used internally by the GoogleGenerationProvider when
preparing tool definitions to be sent to Google's API.

Example:
```python
from agentle.generations.providers.google._adapters.agentle_tool_to_google_tool_adapter import (
    AgentleToolToGoogleToolAdapter
)
from agentle.generations.tools.tool import Tool

# Create an Agentle tool
weather_tool = Tool(
    name="get_weather",
    description="Get the current weather for a location",
    parameters={
        "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA",
            "required": True
        },
        "unit": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"],
            "default": "celsius"
        }
    }
)

# Convert to Google's format
adapter = AgentleToolToGoogleToolAdapter()
google_tool = adapter.adapt(weather_tool)

# Now use with Google's API
response = model.generate_content(
    "What's the weather in London?",
    tools=[google_tool]
)
```
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from rsb.adapters.adapter import Adapter

from agentle.generations.tools.tool import Tool

if TYPE_CHECKING:
    from google.genai import types


class AgentleToolToGoogleToolAdapter(Adapter[Tool[Any], "types.Tool"]):
    """
    Adapter for converting Agentle Tool objects to Google AI Tool format.

    This adapter transforms Agentle's Tool objects into the FunctionDeclaration-based
    Tool format used by Google's Generative AI APIs. It handles the mapping between
    Agentle's parameter definitions and Google's schema-based format, including
    type conversion, required parameters, and default values.

    The adapter implements Agentle's provider abstraction layer pattern, which allows
    tools defined once to be used across different AI providers without modification.

    Key features:
    - Conversion of parameter types from string-based to Google's Type enum
    - Handling of required parameters
    - Support for default values
    - Basic support for array types

    Example:
        ```python
        # Create an Agentle tool for fetching population data
        population_tool = Tool(
            name="get_population",
            description="Get the population of a city",
            parameters={
                "city": {
                    "type": "string",
                    "description": "The name of the city",
                    "required": True
                },
                "country": {
                    "type": "string",
                    "description": "The country of the city",
                    "required": False,
                    "default": "USA"
                }
            }
        )

        # Convert to Google's format
        adapter = AgentleToolToGoogleToolAdapter()
        google_tool = adapter.adapt(population_tool)
        ```
    """

    def __init__(self) -> None:
        """Initialize the adapter with a logger."""
        super().__init__()
        self._logger = logging.getLogger(__name__)

    def adapt(self, agentle_tool: Tool[Any]) -> "types.Tool":
        """
        Convert an Agentle Tool to a Google AI Tool.

        This method transforms an Agentle Tool object into Google's Tool format,
        which consists of FunctionDeclaration objects containing parameter schemas.
        The conversion process involves mapping parameter types, handling required fields,
        and converting any special formats between the two systems.

        Args:
            agentle_tool: The Agentle Tool object to convert. This should be a fully
                defined Tool instance with name, description, and parameters.

        Returns:
            types.Tool: A Google AI Tool object containing one or more function
                declarations that represent the Agentle tool's functionality.

        Implementation details:
            - String-based types in Agentle (e.g., "string", "int") are mapped to
              Google's Type enum values (e.g., Type.STRING, Type.INTEGER)
            - Required parameters are tracked and added to the schema's required list
            - Default values are preserved in the conversion
            - Array types get a simplified items schema (currently defaulting to string items)
            - Unknown types default to Type.OBJECT

        Example:
            ```python
            # Import necessary components
            from google.genai import types

            # Create an Agentle tool
            calculator_tool = Tool(
                name="calculate",
                description="Perform a calculation",
                parameters={
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate",
                        "required": True
                    }
                }
            )

            # Convert to Google format
            adapter = AgentleToolToGoogleToolAdapter()
            google_tool = adapter.adapt(calculator_tool)

            # The resulting tool can be used with Google's API
            from google.genai import GenerativeModel
            model = GenerativeModel("gemini-1.5-pro")
            response = model.generate_content(
                "Calculate 15 + 27",
                tools=[google_tool]
            )
            ```
        """
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
            # Handle different parameter types
            if isinstance(param_info_obj, str):
                # If parameter is a string, treat it as a simple string type parameter
                properties[param_name] = types.Schema(type=types.Type.STRING)
                # Consider string parameters as required by default
                required.append(param_name)
            elif isinstance(param_info_obj, list):
                # If parameter is a list, treat it as an array of strings
                properties[param_name] = types.Schema(
                    type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)
                )
                # Lists are not required by default
            elif isinstance(param_info_obj, dict):
                # Regular dictionary parameter handling
                param_info = cast(dict[str, Any], param_info_obj)

                # Get parameter type
                param_type_str = param_info.get("type", "object")
                google_type = type_mapping.get(
                    str(param_type_str).lower(), types.Type.OBJECT
                )  # Default to OBJECT if type unknown

                # Create parameter schema
                if google_type == types.Type.ARRAY:
                    # Handle array type with items schema
                    # Assume array of strings as default
                    properties[param_name] = types.Schema(
                        type=google_type, items=types.Schema(type=types.Type.STRING)
                    )
                else:
                    # Handle scalar types
                    # Default value must be of the correct type for the parameter
                    if "default" in param_info:
                        properties[param_name] = types.Schema(
                            type=google_type, default=param_info["default"]
                        )
                    else:
                        properties[param_name] = types.Schema(type=google_type)

                if param_info.get("required", False):
                    required.append(param_name)
            else:
                # For any other type, default to string type
                properties[param_name] = types.Schema(type=types.Type.STRING)
                self._logger.warning(
                    f"Unknown parameter type for {param_name}, defaulting to string"
                )

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
