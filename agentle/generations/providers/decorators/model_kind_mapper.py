"""
Decorator module for handling model kind mapping in generation providers.

This module provides a decorator that intercepts calls to generation methods and
maps abstract ModelKind values to provider-specific model identifiers using the
provider's own mapping implementation.
"""

from functools import wraps
from typing import Any, Callable, Protocol, cast, get_args

from agentle.generations.providers.types.model_kind import ModelKind


class _ProviderWithModelMapping(Protocol):
    """Protocol defining the required method for model kind mapping."""

    def map_model_kind_to_provider_model(self, model_kind: ModelKind) -> str:
        """Map a model kind to a provider-specific model identifier."""
        ...


def override_model_kind[F: Callable[..., Any]](func: F) -> F:
    """
    Decorator that maps ModelKind enums to provider-specific model identifiers.

    This decorator intercepts calls to methods that accept a 'model' parameter,
    checks if the model is a ModelKind enum, and if so, uses the provider's
    map_model_kind_to_provider_model method to convert it to a provider-specific
    model identifier string.

    Args:
        func: The function to decorate, typically a create_generation_async method
              in a provider implementation.

    Returns:
        The decorated function that handles ModelKind mapping automatically.
    """

    @wraps(func)
    async def wrapper(
        self: _ProviderWithModelMapping, *args: Any, **kwargs: Any
    ) -> Any:
        # Check if model is provided
        if "model" in kwargs and kwargs["model"] is not None:
            model = kwargs["model"]

            # Check if it's a direct ModelKind enum value
            model_kind_values = get_args(ModelKind)

            # Handle direct string value that matches a ModelKind
            if isinstance(model, str) and model in model_kind_values:
                model_kind = cast(ModelKind, model)
                provider_model = self.map_model_kind_to_provider_model(model_kind)
                kwargs["model"] = provider_model

            # Handle actual ModelKind type value
            elif model in model_kind_values:
                model_kind = cast(ModelKind, model)
                provider_model = self.map_model_kind_to_provider_model(model_kind)
                kwargs["model"] = provider_model

        return await func(self, *args, **kwargs)

    return cast(F, wrapper)
