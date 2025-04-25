from typing import Any


class UnsupportedTypeError(TypeError):
    """Error raised when a Python type cannot be mapped to JSON Schema."""

    def __init__(self, py_type: Any):
        super().__init__(f"Cannot generate JSON Schema for Python type: {py_type}")
        self.py_type = py_type
