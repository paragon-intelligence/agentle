"""
Tracing manager for AI generation providers in the Agentle framework.

This module provides the TracingManager class, which encapsulates the logic for creating,
managing, and completing traces for AI generation activities. It is designed to be used by
all generation providers, ensuring consistent tracing behavior while reducing code duplication.

The tracing manager handles the complex aspects of trace creation, hierarchy management,
error handling, and event flushing, allowing generation providers to focus on their
core responsibilities.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional, cast

from agentle.generations.models.generation.generation_config import GenerationConfig
from agentle.generations.providers.base.generation_provider import GenerationProvider
from agentle.generations.tracing.contracts.stateful_observability_client import (
    StatefulObservabilityClient,
)


class TracingManager:
    """
    Manager for tracing AI generation activities across all providers.

    This class encapsulates the logic for creating, managing, and completing traces,
    providing a consistent interface for all generation providers to use. It handles
    the complexities of trace hierarchy, error handling, and event flushing.

    Attributes:
        tracing_client: The client used for observability and tracing.
        provider_name: The name of the provider using this tracing manager.
    """

    def __init__(
        self,
        *,
        tracing_client: Optional[StatefulObservabilityClient] = None,
        provider: GenerationProvider,
    ) -> None:
        """
        Initialize a new tracing manager.

        Args:
            tracing_client: Optional client for observability and tracing.
            provider_name: The name of the provider using this tracing manager.
        """
        self.tracing_client = tracing_client
        self.provider = provider

    async def setup_trace(
        self,
        *,
        generation_config: GenerationConfig,
        model: str,
        input_data: dict[str, Any],
        is_final_generation: bool = False,
    ) -> tuple[
        Optional[StatefulObservabilityClient], Optional[StatefulObservabilityClient]
    ]:
        """
        Set up tracing for a generation by creating or retrieving a trace.

        This method handles the logic for determining whether to create a new trace
        or reuse an existing one, based on the trace_params in the generation_config.

        Args:
            generation_config: The configuration for the generation.
            model: The model being used for generation.
            input_data: The input data for the generation.
            is_final_generation: Whether this is the final generation in a sequence.

        Returns:
            A tuple containing (trace_client, generation_client) for tracing, both may be None.
        """
        if not self.tracing_client:
            return None, None

        trace_params = generation_config.trace_params
        user_id = trace_params.get("user_id", "anonymous")
        session_id = trace_params.get("session_id")

        # Get or create conversation trace
        trace_client = None
        parent_trace_id = trace_params.get("parent_trace_id")

        # Try to get existing trace or create new one
        try:
            if parent_trace_id:
                # Use existing trace ID with parent_trace_id
                trace_client = await self.tracing_client.trace(
                    name=trace_params.get("name"),
                    user_id=user_id,
                    session_id=session_id,
                )
            else:
                # Create new trace
                trace_name = trace_params.get(
                    "name", f"{self.provider.organization}_{model}_conversation"
                )

                trace_client = await self.tracing_client.trace(
                    name=trace_name,
                    user_id=user_id,
                    session_id=session_id,
                    input=input_data.get(
                        "trace_input",
                        {
                            "model": model,
                            "message_count": input_data.get("message_count", 0),
                            "has_tools": input_data.get("has_tools", False),
                            "has_schema": input_data.get("has_schema", False),
                        },
                    ),
                    metadata={
                        "provider": self.provider.organization,
                        "model": model,
                    },
                )

                # Store trace_id for future calls if not final generation
                if trace_client and not is_final_generation:
                    # Check if trace_client has an id attribute and access it safely
                    trace_id = self._get_trace_id(trace_client)
                    if trace_id:
                        trace_params["parent_trace_id"] = trace_id
        except Exception:
            # Fall back to no tracing if we encounter errors
            trace_client = None

        # Set up generation tracing
        generation_client = None
        if trace_client:
            # Get trace metadata
            trace_metadata: dict[str, Any] = {}
            if "metadata" in trace_params and isinstance(
                trace_params["metadata"], dict
            ):
                trace_metadata = {
                    k: v
                    for k, v in trace_params["metadata"].items()
                    if isinstance(k, str)
                }

            # Create generation
            try:
                generation_name = trace_params.get(
                    "name", f"{self.provider.organization}_{model}_generation"
                )

                # Extract config metadata if available
                config_data = {}
                if generation_config:
                    config_data = {
                        k: v
                        for k, v in generation_config.__dict__.items()
                        if not k.startswith("_") and not callable(v)
                    }

                generation_client = await trace_client.model_generation(
                    provider=self.provider.organization,
                    model=model,
                    input_data=input_data,
                    metadata={
                        "config": config_data,
                        **trace_metadata,
                    },
                    name=generation_name,
                )
            except Exception:
                # Fall back to no generation tracing if errors
                generation_client = None

        return trace_client, generation_client

    def _get_trace_id(self, trace_client: StatefulObservabilityClient) -> Optional[str]:
        """
        Safely get the ID from a trace client.

        Args:
            trace_client: The trace client to get the ID from.

        Returns:
            The trace ID if available, otherwise None.
        """
        # Handle trace clients that use different ID attributes
        if hasattr(trace_client, "id"):
            return cast(str, getattr(trace_client, "id"))
        elif hasattr(trace_client, "trace_id"):
            return cast(str, getattr(trace_client, "trace_id"))
        return None

    async def complete_generation(
        self,
        *,
        generation_client: Optional[StatefulObservabilityClient],
        start_time: datetime,
        output_data: dict[str, Any],
        trace_metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Complete a generation with success."""
        if generation_client:
            # Extract usage data from output_data if present
            usage_details = None
            cost_details = None

            if "usage" in output_data:
                usage = output_data["usage"]
                # Format usage data according to Langfuse's expectations
                usage_details = {
                    "input": usage.get("input_tokens"),
                    "output": usage.get("output_tokens"),
                    "total": usage.get("total_tokens"),
                    "unit": "TOKENS",
                }

                # Calculate costs if we have price information
                # This assumes your provider has price_per_million_tokens_input/output methods
                model = trace_metadata.get("model") if trace_metadata else None
                if (
                    model
                    and hasattr(self, "provider")
                    and hasattr(self.provider, "price_per_million_tokens_input")
                ):
                    input_price = self.provider.price_per_million_tokens_input(model)
                    output_price = self.provider.price_per_million_tokens_output(model)

                    input_tokens = usage.get("input_tokens", 0)
                    output_tokens = usage.get("output_tokens", 0)

                    input_cost = (input_tokens / 1_000_000) * input_price
                    output_cost = (output_tokens / 1_000_000) * output_price

                    cost_details = {
                        "input": input_cost,
                        "output": output_cost,
                        "total": input_cost + output_cost,
                        "currency": "USD",
                    }

            # Remove usage from output data to avoid duplication
            output_data_without_usage = {
                k: v for k, v in output_data.items() if k != "usage"
            }

            # Update generation with proper usage and cost details
            await generation_client.end(
                output=output_data_without_usage,
                metadata=trace_metadata or {},
                usage_details=usage_details,
                cost_details=cost_details,
            )

    async def complete_trace(
        self,
        *,
        trace_client: Optional[StatefulObservabilityClient],
        generation_config: GenerationConfig,
        output_data: dict[str, Any],
        success: bool = True,
    ) -> None:
        """
        Complete a trace with success or error.

        This method handles the logic for properly completing a trace and ensuring
        that all events are flushed to the tracing client.

        Args:
            trace_client: The client for the trace to complete.
            generation_config: The configuration used for generation.
            output_data: The data produced by the generation.
            success: Whether the operation was successful.
        """
        if not trace_client:
            return

        try:
            # Complete the trace
            await trace_client.end(
                output=output_data,
                metadata={"completion_status": "success" if success else "error"},
            )

            # Flush events and clean up
            if self.tracing_client:
                await self.tracing_client.flush()

            # Clean up trace_params
            trace_params = generation_config.trace_params
            if "parent_trace_id" in trace_params:
                del trace_params["parent_trace_id"]
        except Exception:
            # Just continue even if we can't clean up properly
            pass

    async def handle_error(
        self,
        *,
        generation_client: Optional[StatefulObservabilityClient],
        trace_client: Optional[StatefulObservabilityClient],
        generation_config: GenerationConfig,
        start_time: datetime,
        error: Exception,
        trace_metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """
        Handle an error during generation.

        This method records the error with the tracing client and ensures proper cleanup.

        Args:
            generation_client: The client for the generation that failed.
            trace_client: The client for the trace containing the generation.
            generation_config: The configuration used for generation.
            start_time: When the generation started.
            error: The exception that occurred.
            trace_metadata: Additional metadata for the trace.
        """
        error_str = str(error) if error else "Unknown error"

        # Complete generation with error
        if generation_client:
            await generation_client.complete_with_error(
                error=error_str,
                start_time=start_time,
                error_type="Exception",
                metadata=trace_metadata or {},
            )

        # Complete the trace with error
        await self.complete_trace(
            trace_client=trace_client,
            generation_config=generation_config,
            output_data={"error": error_str},
            success=False,
        )
