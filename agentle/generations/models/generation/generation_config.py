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
from typing import Any, Self, cast

from rsb.models.base_model import BaseModel
from rsb.models.field import Field
from rsb.models.model_validator import model_validator

from agentle.generations.models.generation.generation_reasoning import (
    GenerationReasoning,
)
from agentle.generations.models.generation.generation_reasoning_dict import (
    GenerationReasoningDict,
)
from agentle.generations.models.generation.google_generation_config import (
    GoogleGenerationConfig,
)
from agentle.generations.models.generation.google_generation_config_dict import (
    GoogleGenerationConfigDict,
)
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
        reasoning: Provider-agnostic reasoning or thinking configuration.
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
    stop_sequences: list[str] | None = Field(
        default=None,
        description="Strings that stop generation when encountered in model output.",
    )
    seed: int | None = Field(
        default=None,
        description="Best-effort deterministic seed for providers that support seeded generation.",
    )
    presence_penalty: float | None = Field(
        default=None,
        description="Penalty applied to tokens already present in the generated text.",
    )
    frequency_penalty: float | None = Field(
        default=None,
        description="Penalty applied to tokens that repeatedly appear in generated text.",
    )
    logprobs: int | None = Field(
        default=None,
        gt=0,
        description="Number of top candidate token log probabilities to request, when supported.",
    )
    response_logprobs: bool | None = Field(
        default=None,
        description="Whether to return chosen-token log probabilities, when supported.",
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

    reasoning: GenerationReasoning | None = Field(
        default=None,
        description="Provider-agnostic reasoning or thinking configuration. Providers that do not support it may ignore this field.",
    )

    google: GoogleGenerationConfig | None = Field(
        default=None,
        description="Google-specific GenerateContentConfig options.",
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

    @model_validator(mode="after")
    def validate_reasoning(self) -> Self:
        if (
            self.reasoning is not None
            and self.reasoning.effort is not None
            and self.reasoning.max_tokens is not None
        ):
            raise ValueError(
                "Only one of reasoning.effort or reasoning.max_tokens should be set."
            )

        return self

    @property
    def timeout_in_seconds(self) -> float | None:
        return (
            self.timeout / 1000
            # Convertendo de segundos para milissegundos
            if self.timeout
            else self.timeout_s
            if self.timeout_s
            else self.timeout_m * 60
            if self.timeout_m
            else None
        )

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
        new_timeout_s: float | None = None,
        new_timeout_m: float | None = None,
        new_reasoning: GenerationReasoning | GenerationReasoningDict | None = None,
        new_stop_sequences: list[str] | None = None,
        new_seed: int | None = None,
        new_presence_penalty: float | None = None,
        new_frequency_penalty: float | None = None,
        new_logprobs: int | None = None,
        new_response_logprobs: bool | None = None,
        new_google: GoogleGenerationConfig | GoogleGenerationConfigDict | None = None,
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
            stop_sequences=new_stop_sequences
            if new_stop_sequences is not None
            else self.stop_sequences,
            seed=new_seed if new_seed is not None else self.seed,
            presence_penalty=new_presence_penalty
            if new_presence_penalty is not None
            else self.presence_penalty,
            frequency_penalty=new_frequency_penalty
            if new_frequency_penalty is not None
            else self.frequency_penalty,
            logprobs=new_logprobs if new_logprobs is not None else self.logprobs,
            response_logprobs=new_response_logprobs
            if new_response_logprobs is not None
            else self.response_logprobs,
            trace_params=self.trace_params,
            timeout=new_timeout if new_timeout is not None else self.timeout,
            timeout_s=new_timeout_s if new_timeout_s is not None else self.timeout_s,
            timeout_m=new_timeout_m if new_timeout_m is not None else self.timeout_m,
            reasoning=cast(
                Any,
                new_reasoning if new_reasoning is not None else self.reasoning,
            ),
            google=cast(Any, new_google if new_google is not None else self.google),
        )

    class Config:
        arbitrary_types_allowed = True
