"""
Decorators for Generation Provider implementations.

This package contains decorators that can be used with generation provider implementations
to add common functionality like model kind mapping.
"""

from agentle.generations.providers.decorators.model_kind_mapper import (
    override_model_kind,
)

__all__ = ["override_model_kind"]
