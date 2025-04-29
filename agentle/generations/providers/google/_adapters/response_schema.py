"""
Adapter module for converting schema definitions to Google AI Schema format.

This module provides the SchemaToGoogleSchemaAdapter class, which transforms dictionary-based
schema definitions into the Schema objects expected by Google's Generative AI APIs.
This conversion is necessary when using structured output with Google's AI models,
allowing for type-safe parsing of model responses.

The adapter handles the recursive conversion of complex schemas with nested properties,
arrays, and enum values. It supports all the schema features provided by Google's API,
including property ordering, type mapping, and validation constraints.

This adapter is typically used internally by the GoogleGenerationProvider when
preparing structured output schemas to be sent to Google's API.

Example:
```python
from agentle.generations.providers.google._adapters.response_schema import (
    SchemaToGoogleSchemaAdapter
)

# Define a schema as a dictionary
schema_dict = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string",
            "description": "Person's full name"
        },
        "age": {
            "type": "integer",
            "minimum": 0,
            "maximum": 120
        },
        "address": {
            "type": "object",
            "properties": {
                "street": {"type": "string"},
                "city": {"type": "string"},
                "country": {"type": "string"}
            }
        }
    },
    "required": ["name"]
}

# Create an adapter
adapter = SchemaToGoogleSchemaAdapter(property_ordering=True)

# Convert to Google's Schema format
google_schema = adapter.adapt(schema_dict)

# Now use with Google's API for structured output
response = model.generate_content(
    "Generate information about John Doe",
    response_schema=google_schema
)
```
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

from rsb.adapters.adapter import Adapter

if TYPE_CHECKING:
    from google.genai.types import Schema


class SchemaToGoogleSchemaAdapter(Adapter[dict[str, object], "Schema"]):
    """
    Adapter for converting dictionary-based schemas to Google AI Schema objects.

    This adapter transforms schema definitions provided as Python dictionaries into
    the Schema objects required by Google's Generative AI APIs. It supports all schema features including nested objects, arrays, enums, and
    validation constraints.

    The adapter recursively processes complex schemas, converting each property
    and maintaining the hierarchical structure. It also handles special features
    like property ordering, which can influence how Google's models generate
    structured responses.

    Attributes:
        property_ordering (bool | Sequence[str] | None): Controls the ordering of
            properties in object schemas. If True, uses the order from the input
            dictionary. If a sequence, uses that specific ordering. If None,
            no explicit ordering is set.

    Conversion support:
    - Basic properties (title, description, format, pattern, etc.)
    - Type mapping (string to Google's Type enum)
    - Enum values
    - Required fields
    - Property ordering
    - Nested object properties
    - Array item schemas
    - Any-of schemas for alternatives

    Example:
        ```python
        # Create a schema for weather information
        weather_schema = {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "temperature": {"type": "number"},
                "conditions": {
                    "type": "string",
                    "enum": ["sunny", "cloudy", "rainy", "snowy"]
                },
                "forecast": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "day": {"type": "string"},
                            "high": {"type": "number"},
                            "low": {"type": "number"}
                        }
                    }
                }
            },
            "required": ["location", "temperature"]
        }

        # Define a specific property order
        property_order = ["location", "temperature", "conditions", "forecast"]

        # Create the adapter with custom property ordering
        adapter = SchemaToGoogleSchemaAdapter(property_ordering=property_order)

        # Convert to Google's Schema format
        google_schema = adapter.adapt(weather_schema)
        ```
    """

    property_ordering: bool | Sequence[str] | None

    def __init__(self, property_ordering: bool | Sequence[str] | None = None) -> None:
        """
        Initialize the adapter with optional property ordering configuration.

        Args:
            property_ordering: Controls how properties in object schemas are ordered.
                If True, uses the key order from the input dictionary.
                If a sequence of strings, uses that specific ordering.
                If None (default), no explicit ordering is set.

        Note:
            Property ordering can influence how Google's models generate structured
            responses, potentially improving the quality and consistency of the
            generated output.
        """
        super().__init__()
        self.property_ordering = property_ordering

    def adapt(self, _f: dict[str, object]) -> Schema:
        """
        Convert a dictionary-based schema to a Google AI Schema object.

        This method transforms a schema definition provided as a Python dictionary
        into the Schema object required by Google's Generative AI APIs. It handles
        all schema features including nested objects, arrays, enums, and validation
        constraints.

        The method recursively processes complex schemas, ensuring that the entire
        hierarchical structure is properly converted to Google's format.

        Args:
            _f: The dictionary-based schema to convert. This should follow JSON Schema
                conventions with fields like type, properties, items, etc.

        Returns:
            Schema: A Google AI Schema object representing the converted schema,
                ready to be used with Google's API for structured output.

        Example:
            ```python
            # Define a simple schema for a person
            person_schema = {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {
                        "type": "integer",
                        "minimum": 0
                    },
                    "hobbies": {
                        "type": "array",
                        "items": {"type": "string"}
                    }
                }
            }

            # Convert to Google's format
            google_schema = adapter.adapt(person_schema)
            ```
        """
        from google.genai.types import Schema, Type

        # Create the schema instance with all applicable fields
        schema_kwargs: dict[str, object] = {}

        # Map basic properties directly
        for key in [
            "title",
            "description",
            "format",
            "pattern",
            "default",
            "nullable",
            "minimum",
            "maximum",
            "min_length",
            "max_length",
            "min_items",
            "max_items",
            "min_properties",
            "max_properties",
            "example",
        ]:
            if key in _f:
                schema_kwargs[key] = _f[key]

        # Handle type field
        if "type" in _f:
            type_value = _f["type"]
            if isinstance(type_value, str):
                try:
                    schema_kwargs["type"] = Type(type_value)
                except ValueError:
                    # If the string doesn't match a Type enum value, leave it as is
                    schema_kwargs["type"] = type_value
            else:
                schema_kwargs["type"] = type_value

        # Handle enum field
        if "enum" in _f:
            enum_values = _f["enum"]
            if isinstance(enum_values, list):
                # Convert all enum values to strings as per the Schema class
                schema_kwargs["enum"] = [str(value) for value in enum_values]  # type: ignore[reportUnknownArgumentType]

        # Handle required field
        if "required" in _f:
            required_values = _f["required"]
            if isinstance(required_values, list):
                schema_kwargs["required"] = list(required_values)  # type: ignore[reportUnknownArgumentType]

        # Handle property ordering
        if "properties" in _f and self.property_ordering is not None:
            properties = cast(dict[str, dict[str, object]], _f["properties"])
            if isinstance(self.property_ordering, bool) and self.property_ordering:
                # Use the keys from the input dictionary as the order
                schema_kwargs["property_ordering"] = list(properties.keys())
            elif isinstance(self.property_ordering, Sequence):
                # Use the provided ordering
                schema_kwargs["property_ordering"] = list(self.property_ordering)

        # Handle nested schema objects

        # Process properties (for OBJECT type)
        if "properties" in _f:
            properties_dict = cast(dict[str, dict[str, object]], _f["properties"])
            processed_properties: dict[str, Schema] = {}

            for prop_name, prop_schema in properties_dict.items():
                # Recursively convert each property
                processed_properties[prop_name] = self.adapt(prop_schema)

            schema_kwargs["properties"] = processed_properties

        # Process items (for ARRAY type)
        if "items" in _f:
            items_schema = cast(dict[str, object], _f["items"])
            schema_kwargs["items"] = self.adapt(items_schema)

        # Process any_of
        if "any_of" in _f:
            any_of_schemas = cast(list[dict[str, object]], _f["any_of"])
            schema_kwargs["any_of"] = [self.adapt(schema) for schema in any_of_schemas]

        return Schema(**schema_kwargs)  # type: ignore[reportArgumentType]
