from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

from rsb.adapters.adapter import Adapter

if TYPE_CHECKING:
    from google.genai.types import Schema


class SchemaToGoogleSchemaAdapter(Adapter[dict[str, object], "Schema"]):
    property_ordering: bool | Sequence[str] | None

    def __init__(self, property_ordering: bool | Sequence[str] | None = None) -> None:
        super().__init__()
        self.property_ordering = property_ordering

    def adapt(self, _f: dict[str, object]) -> "Schema":
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
