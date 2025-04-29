"""
JSON Schema generation utilities for the Agentle framework.

This module provides tools for generating JSON Schema documents from Python types.
It simplifies the process of creating schema representations that can be used for
validation, documentation, and API specifications.

The primary components are:
- JsonSchemaBuilder: Converts Python types to JSON Schema Draft 7 representations
- UnsupportedTypeError: Exception raised when a type cannot be mapped to JSON Schema

These utilities are particularly useful when working with structured data and
type validation in the Agentle framework, especially for parsing model responses
and defining expected output formats for generation providers.
"""

from .json_schema_builder import JsonSchemaBuilder
from .unsuported_type_error import UnsupportedTypeError

__all__ = ["JsonSchemaBuilder", "UnsupportedTypeError"]
