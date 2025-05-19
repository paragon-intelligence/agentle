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
import re
from typing import TYPE_CHECKING, Any, cast
import json

from rsb.adapters.adapter import Adapter

from agentle.generations.tools.tool import Tool

if TYPE_CHECKING:
    from google.genai import types

# Constants for validation
MAX_FUNCTION_NAME_LENGTH = 64
FUNCTION_NAME_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_\.\-]*$")
MAX_PARAM_NAME_LENGTH = 64
PARAM_NAME_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


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

    def _validate_function_name(self, name: str) -> None:
        """Validate function name according to Google's requirements."""
        if not name:
            raise ValueError("Function name cannot be empty")
        if len(name) > MAX_FUNCTION_NAME_LENGTH:
            raise ValueError(
                f"Function name cannot exceed {MAX_FUNCTION_NAME_LENGTH} characters"
            )
        if not FUNCTION_NAME_PATTERN.match(name):
            raise ValueError(
                "Function name must start with a letter or underscore and contain only "
                + "letters, numbers, underscores, dots, or dashes"
            )

    def _validate_parameter_name(self, name: str) -> None:
        """Validate parameter name according to Google's requirements."""
        if not name:
            raise ValueError("Parameter name cannot be empty")
        if len(name) > MAX_PARAM_NAME_LENGTH:
            raise ValueError(
                f"Parameter name cannot exceed {MAX_PARAM_NAME_LENGTH} characters"
            )
        if not PARAM_NAME_PATTERN.match(name):
            raise ValueError(
                "Parameter name must start with a letter or underscore and contain only "
                + "letters, numbers, or underscores"
            )

    def _get_google_type(self, param_type_str: str, param_name: str) -> "types.Type":
        """Convert Agentle type string to Google Type enum."""
        from google.genai import types

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

        google_type = type_mapping.get(str(param_type_str).lower())
        if google_type is None:
            self._logger.warning(
                f"Unknown parameter type '{param_type_str}' for parameter '{param_name}', "
                + "defaulting to OBJECT type"
            )
            google_type = types.Type.OBJECT

        return google_type

    def _create_array_schema(
        self, param_info: dict[str, Any], param_name: str
    ) -> "types.Schema":
        """Create schema for array type parameters with proper item type handling."""
        from google.genai import types

        # Get item type if specified, default to string
        items_type = param_info.get("items", {}).get("type", "string")
        items_google_type = self._get_google_type(items_type, f"{param_name}[items]")

        return types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(type=items_google_type),
            description=param_info.get("description"),
            min_items=param_info.get("min_items"),
            max_items=param_info.get("max_items"),
        )

    def _create_object_schema(
        self, param_info: dict[str, Any], param_name: str
    ) -> "types.Schema":
        """Create schema for object type parameters with nested properties."""
        from google.genai import types

        properties: dict[str, types.Schema] = {}
        required: list[str] = []

        if "properties" in param_info:
            for prop_name, prop_info in param_info["properties"].items():
                self._validate_parameter_name(prop_name)
                prop_type = prop_info.get("type", "object")
                google_type = self._get_google_type(
                    prop_type, f"{param_name}.{prop_name}"
                )

                if prop_info.get("required", False):
                    required.append(prop_name)

                properties[prop_name] = types.Schema(
                    type=google_type,
                    description=prop_info.get("description"),
                    default=prop_info.get("default"),
                )

        schema = types.Schema(
            type=types.Type.OBJECT,
            properties=properties,
            description=param_info.get("description"),
        )
        if required:
            schema.required = required

        return schema

    def adapt(self, agentle_tool: Tool[Any]) -> "types.Tool":
        """
        Convert an Agentle Tool to a Google AI Tool.

        Args:
            agentle_tool: The Agentle Tool object to convert.

        Returns:
            types.Tool: A Google AI Tool object.

        Raises:
            ValueError: If the tool name or parameters are invalid.
        """
        from google.genai import types

        # Validate function name
        self._validate_function_name(agentle_tool.name)

        properties: dict[str, types.Schema] = {}
        required: list[str] = []

        # Process parameters
        for param_name, param_info_obj in agentle_tool.parameters.items():
            self._validate_parameter_name(param_name)

            # Handle string parameters
            if isinstance(param_info_obj, str):
                properties[param_name] = types.Schema(
                    type=types.Type.STRING,
                    description=f"String parameter: {param_name}",
                )
                required.append(param_name)
                continue

            # Handle list parameters
            if isinstance(param_info_obj, list):
                properties[param_name] = types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.STRING),
                    description=f"Array parameter: {param_name}",
                )
                continue

            # Handle dictionary parameters
            if isinstance(param_info_obj, dict):
                param_info = cast(dict[str, Any], param_info_obj)
                param_type_str = param_info.get("type", "object")
                google_type = self._get_google_type(param_type_str, param_name)

                # Validate description is provided
                if "description" not in param_info:
                    self._logger.warning(
                        f"No description provided for parameter '{param_name}', "
                        + "this may affect model performance"
                    )

                # Handle different types
                if google_type == types.Type.ARRAY:
                    properties[param_name] = self._create_array_schema(
                        param_info, param_name
                    )
                elif google_type == types.Type.OBJECT:
                    properties[param_name] = self._create_object_schema(
                        param_info, param_name
                    )
                else:
                    # Handle scalar types
                    schema = types.Schema(
                        type=google_type,
                        description=param_info.get("description"),
                    )

                    # Handle enums
                    if "enum" in param_info:
                        schema.enum = param_info["enum"]

                    # Handle default value
                    if "default" in param_info:
                        schema.default = param_info["default"]

                    properties[param_name] = schema

                if param_info.get("required", False):
                    required.append(param_name)
                continue

            # Handle other types by inferring from Python type
            if isinstance(param_info_obj, (int, float, bool)):
                if isinstance(param_info_obj, bool):
                    properties[param_name] = types.Schema(type=types.Type.BOOLEAN)
                elif isinstance(param_info_obj, int):
                    properties[param_name] = types.Schema(type=types.Type.INTEGER)
                else:
                    properties[param_name] = types.Schema(type=types.Type.NUMBER)
                self._logger.info(f"Inferred type for {param_name} from Python type")
                continue

            # Handle collection types
            if isinstance(param_info_obj, (set, tuple, frozenset)):
                properties[param_name] = types.Schema(
                    type=types.Type.ARRAY,
                    items=types.Schema(type=types.Type.STRING),
                )
                self._logger.info(f"Inferred array type for {param_name}")
                continue

            # Try to handle complex types
            try:
                json.dumps(param_info_obj)
                properties[param_name] = types.Schema(type=types.Type.OBJECT)
                self._logger.info(f"Inferred object type for {param_name}")
            except (TypeError, OverflowError, ValueError):
                properties[param_name] = types.Schema(type=types.Type.STRING)
                self._logger.warning(
                    f"Unknown parameter type for {param_name}, defaulting to string"
                )

        # Create parameters schema
        parameters_schema = types.Schema(type=types.Type.OBJECT, properties=properties)
        if required:
            parameters_schema.required = required

        # Create function declaration
        function_declaration = types.FunctionDeclaration(
            name=agentle_tool.name,
            description=agentle_tool.description or "",
            parameters=parameters_schema,
        )

        # Create and return tool
        return types.Tool(function_declarations=[function_declaration])
