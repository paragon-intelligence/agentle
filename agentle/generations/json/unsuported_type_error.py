"""
Exception class for handling unsupported types in JSON Schema generation.

This module defines the exception raised when the JsonSchemaBuilder encounters
a Python type that cannot be converted to a valid JSON Schema representation.
This exception helps distinguish type conversion errors from other exceptions
that might occur during schema generation.
"""

from typing import Any


class UnsupportedTypeError(TypeError):
    """
    Error raised when a Python type cannot be mapped to JSON Schema.

    This exception is thrown by the JsonSchemaBuilder when it encounters a Python type
    that has no direct mapping to JSON Schema, or when the type is too complex or
    ambiguous to be automatically converted. Examples include certain metaclasses,
    callables, unsupported generics, or types with unresolvable forward references.
    """

    def __init__(self, py_type: Any):
        """
        Initialize the exception with the unsupported Python type.

        Args:
            py_type: The Python type that could not be converted to JSON Schema.
                This can be a type object, a string representation of a type,
                or any other value that couldn't be processed.
        """
        super().__init__(f"Cannot generate JSON Schema for Python type: {py_type}")
        self.py_type = py_type
