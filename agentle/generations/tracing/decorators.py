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
from collections.abc import Callable, Coroutine
from datetime import datetime
from typing import Any, TypeVar, cast
from rsb.coroutines.fire_and_forget import fire_and_forget
from agentle.generations.models.generation.generation import Generation
from agentle.generations.models.generation.generation_config import GenerationConfig
from agentle.generations.providers.base.generation_provider import GenerationProvider
from agentle.generations.tracing.contracts.stateful_observability_client import (
    StatefulObservabilityClient,
)
from agentle.generations.tracing.tracing_manager import TracingManager

T = TypeVar('T')
P = TypeVar('P', bound=dict[str, Any])

logger = logging.getLogger(__name__)

def observe(
    func: Callable[..., Coroutine[Any, Any, Generation[Any]]],
) -> Callable[..., Coroutine[Any, Any, Generation[Any]]]:
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
            await tracing_manager.complete_generation(
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
                    await trace_client.score_trace(
                        name="trace_success",
                        value=1.0,
                        comment="Generation completed successfully"
                    )
                except Exception as e:
                    logger.warning(f"Failed to add trace success score: {e}")

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
                    await trace_client.score_trace(
                        name="trace_success",
                        value=0.0,
                        comment=f"Error: {error_type} - {error_str[:100]}"
                    )
                except Exception as scoring_error:
                    logger.warning(f"Failed to add trace error score: {scoring_error}")
            
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

    return wrapper
