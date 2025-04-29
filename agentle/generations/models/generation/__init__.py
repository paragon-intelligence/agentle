"""
Core generation models for the Agentle framework.

This module provides the core data structures used to represent AI model generations
and their associated metadata throughout the Agentle framework. These models form
the foundation of Agentle's provider-agnostic interface, ensuring consistent
handling of AI responses regardless of the underlying model provider.

The module includes:
- Generation: Primary container for model outputs with metadata and usage statistics
- Choice: Individual candidate response from a model
- GenerationConfig: Configuration parameters for generation requests
- Usage: Token usage tracking for prompts and completions
- TraceParams: Optional parameters for tracing and logging generations

These models are designed to be provider-independent, allowing applications to
work with different AI providers through a unified interface while maintaining
access to all relevant metadata and configurations.
"""

from .choice import Choice
from .generation import Generation
from .generation_config import GenerationConfig
from .trace_params import TraceParams
from .usage import Usage

__all__ = ["Choice", "Generation", "GenerationConfig", "TraceParams", "Usage"]
