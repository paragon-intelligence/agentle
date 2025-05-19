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
from typing import TYPE_CHECKING, Any, Mapping, TypedDict, cast

from rsb.adapters.adapter import Adapter

from agentle.generations.tools.tool import Tool

if TYPE_CHECKING:
    from google.genai import types

# Constants for validation
MAX_FUNCTION_NAME_LENGTH = 64
FUNCTION_NAME_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_\.\-]*$")
MAX_PARAM_NAME_LENGTH = 64
PARAM_NAME_PATTERN = re.compile(r"^[a-zA-Z_$][a-zA-Z0-9_]*$")

# Special JSON Schema keywords that should be allowed despite not matching the normal pattern
JSON_SCHEMA_KEYWORDS = {"$schema", "$ref", "$id", "$defs", "$comment", "$vocabulary"}

# Type aliases for better readability
JSONValue = bool | int | float | str | list[Any] | dict[str, Any] | None
JSONObject = dict[str, JSONValue]


class JSONSchemaDict(TypedDict, total=False):
    """Type definition for JSON Schema dictionary."""

    type: str
    description: str | None
    default: Any
    properties: dict[str, "JSONSchemaDict"]
    items: "JSONSchemaDict" | list["JSONSchemaDict"]
    required: list[str]
    minItems: int | None
    maxItems: int | None
    minLength: int | None
    maxLength: int | None
    pattern: str | None
    minimum: float | None
    maximum: float | None
    enum: list[Any] | None
    additionalProperties: bool | None
    schema: str | None  # Using schema instead of $schema due to Python syntax


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

        # Allow special JSON Schema keywords
        if name in JSON_SCHEMA_KEYWORDS:
            return

        if not PARAM_NAME_PATTERN.match(name):
            raise ValueError(
                "Parameter name must start with a letter, underscore, or $ and contain only letters, numbers, or underscores"
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

    def _create_schema_from_json_schema(
        self, schema_dict: Mapping[str, Any], param_name: str = ""
    ) -> "types.Schema":
        """Create a Google Schema from a JSON Schema definition."""
        from google.genai import types

        # Get the type
        schema_type = str(schema_dict.get("type", "object"))
        google_type = self._get_google_type(schema_type, param_name)

        # Create base schema
        schema = types.Schema(
            type=google_type,
            description=str(schema_dict.get("description"))
            if schema_dict.get("description")
            else None,
            default=schema_dict.get("default"),
        )

        # Handle array type
        if google_type == types.Type.ARRAY:
            items_schema = schema_dict.get("items", {})
            if isinstance(items_schema, dict):
                schema.items = self._create_schema_from_json_schema(
                    cast(Mapping[str, Any], items_schema), f"{param_name}[items]"
                )
            else:
                # Default to string items if items schema is not an object
                schema.items = types.Schema(type=types.Type.STRING)

            # Add array constraints
            if "minItems" in schema_dict:
                schema.min_items = int(schema_dict["minItems"])
            if "maxItems" in schema_dict:
                schema.max_items = int(schema_dict["maxItems"])

        # Handle object type
        elif google_type == types.Type.OBJECT:
            properties = schema_dict.get("properties", {})
            schema_properties: dict[str, types.Schema] = {}

            for prop_name, prop_schema in properties.items():
                if not isinstance(prop_schema, dict):
                    continue
                schema_properties[prop_name] = self._create_schema_from_json_schema(
                    cast(Mapping[str, Any], prop_schema),
                    f"{param_name}.{prop_name}" if param_name else prop_name,
                )

            schema.properties = schema_properties

            # Handle required properties
            if "required" in schema_dict:
                schema.required = list(schema_dict["required"])

        # Handle string type
        elif google_type == types.Type.STRING:
            if "minLength" in schema_dict:
                schema.min_length = int(schema_dict["minLength"])
            if "maxLength" in schema_dict:
                schema.max_length = int(schema_dict["maxLength"])
            if "pattern" in schema_dict:
                schema.pattern = str(schema_dict["pattern"])

        # Handle number/integer type
        elif google_type in (types.Type.NUMBER, types.Type.INTEGER):
            if "minimum" in schema_dict:
                schema.minimum = float(schema_dict["minimum"])
            if "maximum" in schema_dict:
                schema.maximum = float(schema_dict["maximum"])

        # Handle enums for any type
        if "enum" in schema_dict:
            schema.enum = list(schema_dict["enum"])

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

        # Convert parameters
        parameters_schema = None
        if agentle_tool.parameters:
            parameters_schema = self._create_schema_from_json_schema(
                agentle_tool.parameters
            )

        # Create function declaration
        function_declaration = types.FunctionDeclaration(
            name=agentle_tool.name,
            description=agentle_tool.description or "",
            parameters=parameters_schema,
        )

        # Create and return tool
        return types.Tool(function_declarations=[function_declaration])
