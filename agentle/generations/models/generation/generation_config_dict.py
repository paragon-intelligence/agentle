"""
Configuration parameters for AI generation requests in the Agentle framework.

This module defines the GenerationConfig class, which encapsulates the various
parameters that can be used to control AI generation behavior. These parameters
include common settings like temperature and top_p that are supported across
many AI providers, as well as provider-specific settings.

The configuration provides a standardized way to specify generation parameters
regardless of which underlying AI provider is being used, allowing for consistent
behavior and easy switching between providers.
"""

from __future__ import annotations

from typing import NotRequired, TypedDict

from agentle.generations.models.generation.trace_params import TraceParams


class GenerationConfigDict(TypedDict):
    """
    Configuration parameters for controlling AI generation behavior.

    This class defines the various parameters that can be adjusted to control
    how AI models generate text. It includes common parameters supported across
    different providers (like temperature and top_p), as well as settings for
    tracing, timeouts, and provider-specific options.

    Attributes:
        temperature: Controls randomness in generation. Higher values (e.g., 0.8) make output
            more random, lower values (e.g., 0.2) make it more deterministic. Range 0-1.
        max_output_tokens: Maximum number of tokens to generate in the response.
        n: Number of alternative completions to generate.
        top_p: Nucleus sampling parameter - considers only the top p% of probability mass.
            Range 0-1.
        top_k: Only sample from the top k tokens at each step.
        trace_params: Parameters for tracing the generation for observability.
        timeout: Maximum time in seconds to wait for a generation before timing out.
    """

    temperature: NotRequired[float | None]

    max_output_tokens: NotRequired[int | None]

    n: NotRequired[int]

    top_p: NotRequired[float | None]

    top_k: NotRequired[float | None]

    trace_params: NotRequired[TraceParams]

    timeout: NotRequired[float | None]

    timeout_s: NotRequired[float | None]

    timeout_m: NotRequired[float | None]
