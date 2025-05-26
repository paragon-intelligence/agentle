"""
Decorators for simplifying observability and tracing integration with generation providers.

This module provides decorators that can be applied to provider methods to automatically
handle observability concerns like tracing, error handling, and metric collection.
The main decorator, `observe`, abstracts away the boilerplate code for setting up traces,
capturing metrics, and properly handling errors with observability in mind.
"""

from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any, TypeVar, cast, get_args
from rsb.coroutines.fire_and_forget import fire_and_forget
from agentle.generations.models.generation.generation import Generation
from agentle.generations.models.generation.generation_config import GenerationConfig
from agentle.generations.providers.base.generation_provider import GenerationProvider
from agentle.generations.providers.types.model_kind import ModelKind
from agentle.generations.tracing.contracts.stateful_observability_client import (
    StatefulObservabilityClient,
)
from agentle.generations.tracing.tracing_manager import TracingManager

T = TypeVar("T")
P = TypeVar("P", bound=dict[str, Any])

logger = logging.getLogger(__name__)


def observe[F: Callable[..., Any]](
    func: F,
) -> F:
    """
    Decorator that adds observability to provider generation methods.

    This decorator wraps generation methods (like create_generation_async) to automatically
    handle observability concerns such as trace creation, metric collection, and error handling.

    When applied to a method, it:
    1. Sets up appropriate traces before execution
    2. Collects execution metrics (latency, token usage)
    3. Handles proper error tracing
    4. Ensures traces are properly completed

    The decorated method can focus purely on the generation logic while observability
    is handled transparently by this decorator.

    Usage:
        ```python
        class MyProvider(GenerationProvider):
            @observe
            async def create_generation_async(self, ...) -> Generation[T]:
                # Method can now focus purely on generation logic
                # without manual observability code
        ```

    Args:
        func: The generation method to decorate

    Returns:
        A wrapped function that handles observability automatically
    """

    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Generation[Any]:
        # Get the provider instance (self)
        provider_self = args[0]

        # Ensure we're decorating a method on a GenerationProvider
        if not isinstance(provider_self, GenerationProvider):
            raise TypeError(
                "The @observe decorator can only be used on methods of GenerationProvider classes"
            )

        # Create a tracing manager if not already present
        tracing_manager = getattr(provider_self, "tracing_manager", None)
        if tracing_manager is None:
            tracing_client = cast(
                StatefulObservabilityClient | None,
                getattr(provider_self, "tracing_client", None),
            )
            tracing_manager = TracingManager(
                tracing_client=tracing_client,
                provider=provider_self,
            )

        # Get parameter values
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Extract the parameters we need for tracing
        model = bound_args.arguments.get("model") or provider_self.default_model

        # Check if model is a ModelKind and map it to provider model if needed
        model_kind_values = get_args(ModelKind)

        if model in model_kind_values:
            # Cast to ModelKind and map to actual provider model
            model_kind = cast(ModelKind, model)
            model = provider_self.map_model_kind_to_provider_model(model_kind)

        messages = bound_args.arguments.get("messages", [])
        response_schema = bound_args.arguments.get("response_schema")
        generation_config = (
            bound_args.arguments.get("generation_config") or GenerationConfig()
        )
        tools = bound_args.arguments.get("tools")

        # Prepare input data for tracing
        input_data: dict[str, Any] = {
            "messages": [
                {
                    "role": msg.role,
                    "content": "".join(str(part) for part in msg.parts),
                }
                for msg in messages
            ],
            "response_schema": str(response_schema) if response_schema else None,
            "tools_count": len(tools) if tools else 0,
            "message_count": len(messages),
            "has_tools": tools is not None and len(tools) > 0,
            "has_schema": response_schema is not None,
        }

        # Add any generation config parameters if available
        if hasattr(generation_config, "__dict__"):
            for key, value in generation_config.__dict__.items():
                if not key.startswith("_") and not callable(value):
                    input_data[key] = value

        # Set up tracing using the tracing manager
        trace_client, generation_client = await tracing_manager.setup_trace(
            generation_config=generation_config,
            model=model,
            input_data=input_data,
        )

        # Extract trace metadata if available
        trace_metadata: dict[str, Any] = {
            "model": model,  # Ensure model is in metadata for cost calculation
            "provider": provider_self.organization,
        }

        trace_params = generation_config.trace_params
        if "metadata" in trace_params:
            metadata_val = trace_params["metadata"]
            if isinstance(metadata_val, dict):
                # Convert to properly typed dict
                for k, v in metadata_val.items():
                    if isinstance(k, str):
                        trace_metadata[k] = v

        # Track execution time
        start_time = datetime.now()

        try:
            # Execute the actual generation method
            response = await func(*args, **kwargs)

            # Extract usage details from the response if available
            usage_details = None
            usage = getattr(response, "usage", None)
            if usage is not None:
                prompt_tokens = getattr(usage, "prompt_tokens", 0)
                completion_tokens = getattr(usage, "completion_tokens", 0)
                total_tokens = getattr(
                    usage, "total_tokens", prompt_tokens + completion_tokens
                )

                usage_details = {
                    "input": prompt_tokens,
                    "output": completion_tokens,
                    "total": total_tokens,
                    "unit": "TOKENS",
                }

            # Calculate cost details if we have usage info
            cost_details = None
            if usage_details:
                input_tokens = int(usage_details.get("input", 0))
                output_tokens = int(usage_details.get("output", 0))

                input_cost = provider_self.price_per_million_tokens_input(
                    model, input_tokens
                ) * (input_tokens / 1_000_000)

                output_cost = provider_self.price_per_million_tokens_output(
                    model, output_tokens
                ) * (output_tokens / 1_000_000)

                cost_details = {
                    "input": input_cost,
                    "output": output_cost,
                    "total": input_cost + output_cost,
                }

            # Prepare output data for tracing
            output_data: dict[str, Any] = {
                "completion": getattr(response, "text", str(response)),
            }

            # Add usage and cost details if available
            if usage_details:
                output_data["usage"] = usage_details
            if cost_details:
                output_data["cost_details"] = cost_details

            # Complete the generation and trace if needed
            fire_and_forget(
                tracing_manager.complete_generation,
                generation_client=generation_client,
                start_time=start_time,
                output_data=output_data,
                trace_metadata=trace_metadata,
                usage_details=usage_details,
                cost_details=cost_details,
            )

            parsed = getattr(response, "parsed", None)
            text = getattr(response, "text", str(response))

            # Add trace success score when the operation completes successfully
            if trace_client:
                try:
                    # Success score (already implemented)
                    await trace_client.score_trace(
                        name="trace_success",
                        value=1.0,
                        comment="Generation completed successfully",
                    )

                    # Calculate latency score based on response time
                    latency_seconds = (datetime.now() - start_time).total_seconds()
                    latency_score = 0.0
                    if latency_seconds < 1.0:
                        latency_score = 1.0  # Excellent (sub-second)
                    elif latency_seconds < 3.0:
                        latency_score = 0.8  # Good (1-3 seconds)
                    elif latency_seconds < 6.0:
                        latency_score = 0.6  # Acceptable (3-6 seconds)
                    elif latency_seconds < 10.0:
                        latency_score = 0.4  # Slow (6-10 seconds)
                    else:
                        latency_score = 0.2  # Very slow (>10 seconds)

                    await trace_client.score_trace(
                        name="latency_score",
                        value=latency_score,
                        comment=f"Response time: {latency_seconds:.2f}s",
                    )

                    # Model tier score - categorize models by capability level
                    model_tier = 0.5  # Default for basic models
                    model_name = model.lower()

                    # Advanced models get higher scores
                    if any(
                        premium in model_name
                        for premium in [
                            "gpt-4",
                            "claude-3-opus",
                            "claude-3-sonnet",
                            "gemini-1.5-pro",
                            "gemini-2.0-pro",
                            "claude-3-7",
                        ]
                    ):
                        model_tier = 1.0
                    elif any(
                        mid in model_name
                        for mid in [
                            "gemini-1.5-flash",
                            "gemini-2.0-flash",
                            "claude-3-haiku",
                            "gpt-3.5",
                        ]
                    ):
                        model_tier = 0.7

                    await trace_client.score_trace(
                        name="model_tier",
                        value=model_tier,
                        comment=f"Model capability tier: {model}",
                    )

                    # Add tool usage score if tools were provided
                    if tools is not None and len(tools) > 0:
                        # Use the Generation's API to access tool calls
                        tool_calls = response.tool_calls

                        # Score based on whether tools were used when available
                        tool_usage_score = 0.0
                        if tool_calls and len(tool_calls) > 0:
                            tool_usage_score = 1.0  # Tools were used
                            tool_comment = (
                                f"Tools were used ({len(tool_calls)} function calls)"
                            )
                        else:
                            # Tools were available but not used
                            tool_usage_score = 0.0
                            tool_comment = "Tools were available but not used"

                        await trace_client.score_trace(
                            name="tool_usage",
                            value=tool_usage_score,
                            comment=tool_comment,
                        )

                    # Token efficiency and cost scores (when token data is available)
                    if usage_details:
                        input_tokens = int(usage_details.get("input", 0))
                        output_tokens = int(usage_details.get("output", 0))

                        # Only calculate if we have meaningful token counts
                        if input_tokens > 0 and output_tokens > 0:
                            # Token efficiency score (balance between input and useful output)
                            # A ratio of output to input between 0.3 and 0.7 is generally good
                            # Too low means little output for input, too high might mean verbose output
                            ratio = min(
                                1.0, float(output_tokens) / max(1, float(input_tokens))
                            )
                            efficiency_score = 0.0

                            if 0.2 <= ratio <= 0.8:
                                # Ideal range gets highest score
                                efficiency_score = 1.0 - abs(0.5 - ratio)
                            else:
                                # Outside ideal range gets lower scores
                                efficiency_score = max(0.0, 0.5 - abs(0.5 - ratio))

                            await trace_client.score_trace(
                                name="token_efficiency",
                                value=efficiency_score,
                                comment=f"Token ratio (output/input): {ratio:.2f}",
                            )

                        # Cost efficiency score (when cost details are available)
                        if cost_details:
                            total_cost = float(cost_details.get("total", 0))
                            if total_cost > 0:
                                # Score inversely proportional to cost
                                # Lower cost is better, normalize to a reasonable range
                                # Cost thresholds in USD
                                cost_score = 0.0
                                if total_cost < 0.001:
                                    cost_score = 1.0  # Very inexpensive (<$0.001)
                                elif total_cost < 0.01:
                                    cost_score = 0.8  # Inexpensive ($0.001-$0.01)
                                elif total_cost < 0.05:
                                    cost_score = 0.6  # Moderate ($0.01-$0.05)
                                elif total_cost < 0.1:
                                    cost_score = 0.4  # Expensive ($0.05-$0.1)
                                else:
                                    cost_score = 0.2  # Very expensive (>$0.1)

                                await trace_client.score_trace(
                                    name="cost_efficiency",
                                    value=cost_score,
                                    comment=f"Cost: ${total_cost:.4f}",
                                )
                except Exception as e:
                    logger.warning(f"Failed to add trace scores: {e}")

            fire_and_forget(
                tracing_manager.complete_trace,
                trace_client=trace_client,
                generation_config=generation_config,
                output_data=parsed or text,
                success=True,
            )

            return response

        except Exception as e:
            # Add trace error score
            if trace_client:
                try:
                    error_type = type(e).__name__
                    error_str = str(e)

                    # Main trace success score (already implemented)
                    await trace_client.score_trace(
                        name="trace_success",
                        value=0.0,
                        comment=f"Error: {error_type} - {error_str[:100]}",
                    )

                    # Add error category score for better filtering
                    error_category = "other"

                    # Categorize common error types
                    if "timeout" in error_str.lower() or "time" in error_type.lower():
                        error_category = "timeout"
                    elif (
                        "connection" in error_str.lower()
                        or "network" in error_str.lower()
                    ):
                        error_category = "network"
                    elif (
                        "auth" in error_str.lower()
                        or "key" in error_str.lower()
                        or "credential" in error_str.lower()
                    ):
                        error_category = "authentication"
                    elif (
                        "limit" in error_str.lower()
                        or "quota" in error_str.lower()
                        or "rate" in error_str.lower()
                    ):
                        error_category = "rate_limit"
                    elif (
                        "value" in error_type.lower()
                        or "type" in error_type.lower()
                        or "attribute" in error_type.lower()
                    ):
                        error_category = "validation"
                    elif (
                        "memory" in error_str.lower() or "resource" in error_str.lower()
                    ):
                        error_category = "resource"

                    await trace_client.score_trace(
                        name="error_category",
                        value=error_category,
                        comment=f"Error classified as: {error_category}",
                    )

                    # Add error severity score
                    severity = 0.7  # Default medium-high severity

                    # Adjust severity based on error type
                    if error_category in ["timeout", "network", "rate_limit"]:
                        # Transient errors - lower severity
                        severity = 0.5
                    elif error_category in ["authentication", "validation"]:
                        # Configuration/code errors - higher severity
                        severity = 0.9

                    await trace_client.score_trace(
                        name="error_severity",
                        value=severity,
                        comment=f"Error severity: {severity:.1f}",
                    )

                    # Calculate latency until error
                    error_latency = (datetime.now() - start_time).total_seconds()
                    await trace_client.score_trace(
                        name="error_latency",
                        value=error_latency,
                        comment=f"Time until error: {error_latency:.2f}s",
                    )
                except Exception as scoring_error:
                    logger.warning(f"Failed to add trace error scores: {scoring_error}")

            # Handle errors using the tracing manager
            await tracing_manager.handle_error(
                generation_client=generation_client,
                trace_client=trace_client,
                generation_config=generation_config,
                start_time=start_time,
                error=e,
                trace_metadata=trace_metadata,
            )
            # Re-raise the exception
            raise

    return cast(F, wrapper)
