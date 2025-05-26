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
from typing import Self

from rsb.models.base_model import BaseModel
from rsb.models.field import Field
from rsb.models.model_validator import model_validator

from agentle.generations.models.generation.trace_params import TraceParams


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

    trace_params: TraceParams = Field(
        default_factory=lambda: TraceParams(),
        description="Configuration for tracing and observability of the generation process. Controls what metadata is captured about the generation for monitoring, debugging, and analysis purposes.",
    )

    timeout: float | None = Field(
        default=None,
        description="Maximum time in milliseconds to wait for a generation response before timing out. Helps prevent indefinite waits for slow or stuck generations. Recommended to set based on expected model and prompt complexity.",
        gt=0,
        examples=[10000, 30000, 60000],
    )

    timeout_s: float | None = Field(
        default=None,
        description="Maximum time in seconds to wait for a generation response before timing out. Helps prevent indefinite waits for slow or stuck generations. Recommended to set based on expected model and prompt complexity.",
        gt=0,
        examples=[10.0, 30.0, 60.0],
    )

    timeout_m: float | None = Field(
        default=None,
        description="Maximum time in minutes to wait for a generation response before timing out. Helps prevent indefinite waits for slow or stuck generations. Recommended to set based on expected model and prompt complexity.",
        gt=0,
        examples=[1.0, 3.0, 6.0],
    )

    @model_validator(mode="after")
    def validate_timeout(self) -> Self:
        # check if all timeout fields are set. only one of them should be set.
        if (
            self.timeout is not None
            and self.timeout_s is not None
            and self.timeout_m is not None
        ):
            raise ValueError(
                "Only one of timeout or timeout_s or timeout_m should be set."
            )

        return self

    def clone(
        self,
        *,
        new_temperature: float | None = None,
        new_max_output_tokens: int | None = None,
        new_n: int | None = None,
        new_top_p: float | None = None,
        new_top_k: float | None = None,
        new_trace_params: TraceParams | None = None,
        new_timeout: float | None = None,
    ) -> GenerationConfig:
        """
        Creates a new GenerationConfig with optionally updated parameters.

        This method allows creating a modified copy of the current configuration
        without altering the original object, following the immutable pattern.

        Args:
            new_temperature: New temperature value, if provided.
            new_max_output_tokens: New maximum output tokens value, if provided.
            new_n: New number of completions value, if provided.
            new_top_p: New top_p value, if provided.
            new_top_k: New top_k value, if provided.
            new_trace_params: New trace parameters, if provided.
            new_timeout: New timeout value, if provided.

        Returns:
            A new GenerationConfig instance with the specified updates applied.
        """

        if new_trace_params is not None:
            self.trace_params.update(new_trace_params)

        return GenerationConfig(
            temperature=new_temperature
            if new_temperature is not None
            else self.temperature,
            max_output_tokens=new_max_output_tokens
            if new_max_output_tokens is not None
            else self.max_output_tokens,
            n=new_n if new_n is not None else self.n,
            top_p=new_top_p if new_top_p is not None else self.top_p,
            top_k=new_top_k if new_top_k is not None else self.top_k,
            trace_params=self.trace_params,
            timeout=new_timeout if new_timeout is not None else self.timeout,
        )

    class Config:
        arbitrary_types_allowed = True
