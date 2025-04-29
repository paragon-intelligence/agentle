"""
Cerebras AI provider implementation for the Agentle framework.

This module provides integration with Cerebras AI services, allowing Agentle
to use Cerebras language models through a consistent interface. Cerebras is
known for its Wafer-Scale Engine and AI accelerator technology.

The module implements the necessary provider interfaces to maintain compatibility
with the broader Agentle framework while handling all Cerebras-specific details
internally. It supports message-based interactions, structured output parsing,
and maintains consistent error handling and response formats.

This provider is part of Agentle's provider abstraction layer, which allows
applications to work with multiple AI providers interchangeably.
"""

from .cerebras_generation_provider import CerebrasGenerationProvider

__all__ = ["CerebrasGenerationProvider"]
