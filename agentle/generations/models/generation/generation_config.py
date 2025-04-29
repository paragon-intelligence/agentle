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

from agentle.generations.models.generation.trace_params import TraceParams
from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class GenerationConfig(BaseModel):
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
        google_generation_kwargs: Additional parameters specific to Google AI models.
        trace_params: Parameters for tracing the generation for observability.
        timeout: Maximum time in seconds to wait for a generation before timing out.
    """

    temperature: float | None = Field(
        default=None,
        description="Controls randomness in text generation. Higher values (e.g., 0.8) produce more diverse and creative outputs, while lower values (e.g., 0.2) produce more focused and deterministic results. Setting to 0 means deterministic output.",
        ge=0.0,
        le=1.0,
        examples=[0.0, 0.5, 0.7, 1.0],
    )
    max_output_tokens: int | None = Field(
        default=None,
        description="Maximum number of tokens the model will generate in its response. Helps control response length and prevent excessively long outputs. Setting too low may truncate important information.",
        gt=0,
        examples=[256, 1024, 4096],
    )
    n: int = Field(
        default=1,
        description="Number of alternative completions to generate for the same prompt. Useful for providing different response options or for techniques like self-consistency that require multiple generations.",
        ge=1,
        examples=[1, 3, 5],
    )
    top_p: float | None = Field(
        default=None,
        description="Nucleus sampling parameter that controls diversity by considering tokens comprising the top_p probability mass. A value of 0.9 means only considering tokens in the top 90% of probability mass. Lower values increase focus, higher values increase diversity.",
        ge=0.0,
        le=1.0,
        examples=[0.9, 0.95, 1.0],
    )
    top_k: float | None = Field(
        default=None,
        description="Limits token selection to the top k most likely tokens at each generation step. Helps filter out low-probability tokens. Lower values restrict creativity but increase focus and coherence.",
        ge=0.0,
        examples=[10, 40, 100],
    )
    google_generation_kwargs: dict[str, object] | None = Field(
        default=None,
        description="Additional parameters specific to Google AI model generation. Allows passing provider-specific parameters that aren't standardized across all providers in the Agentle framework.",
    )
    trace_params: TraceParams = Field(
        default_factory=lambda: TraceParams(),
        description="Configuration for tracing and observability of the generation process. Controls what metadata is captured about the generation for monitoring, debugging, and analysis purposes.",
    )
    timeout: float | None = Field(
        default=None,
        description="Maximum time in seconds to wait for a generation response before timing out. Helps prevent indefinite waits for slow or stuck generations. Recommended to set based on expected model and prompt complexity.",
        gt=0,
        examples=[10.0, 30.0, 60.0],
    )

    class Config:
        arbitrary_types_allowed = True
