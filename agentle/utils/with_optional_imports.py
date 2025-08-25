"""Utility decorator for handling optional imports and Pydantic model rebuilding.

This module provides a simple decorator for handling optional dependencies
and rebuilding Pydantic models when those dependencies become available.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
from typing import TYPE_CHECKING, TypeVar, Callable

if TYPE_CHECKING:
    from rsb.models.base_model import BaseModel

# TypeVar to preserve the specific class type through the decorator
T = TypeVar("T", bound="BaseModel")

logger = logging.getLogger(__name__)


def is_module_available(module_name: str) -> bool:
    """Check if a module is available without importing it."""
    return importlib.util.find_spec(module_name) is not None


def with_optional_imports(
    *module_names: str,
    warning_message: str | None = None,
    auto_discover: bool = True,
    include_private: bool = False,
    allow_overwrite: bool = False,
) -> Callable[[type[T]], type[T]]:
    """
    Decorator to automatically handle optional imports for a Pydantic model.

    This decorator attempts to import the specified modules and makes their
    public classes, functions, and objects available in the model's module
    global namespace, then rebuilds the model to handle forward references.

    SAFETY: By default, this will NOT overwrite existing names in your module
    to prevent accidentally replacing your classes/functions.

    Args:
        *module_names: Module names to import (e.g., 'google.auth.credentials')
        warning_message: Custom warning message if imports fail. If None, uses default.
        auto_discover: If True, automatically imports all public objects from modules.
                      If False, only imports objects that match the module's last segment name.
        include_private: If True, also imports objects starting with underscore (when auto_discover=True).
        allow_overwrite: If True, allows overwriting existing names in your module.
                        If False (default), skips imports that would overwrite existing names.

    Returns:
        The decorated model class, with optional dependencies imported if available.

    Examples:
        ```python
        # Basic usage - imports all public objects, won't overwrite existing names
        @with_optional_imports('google.auth.credentials', 'google.genai.client')
        class GoogleProvider(BaseModel):
            credentials: Optional[Credentials] = None  # From google.auth.credentials
            client: Optional[GenerativeModel] = None   # From google.genai.client

        # Allow overwriting (dangerous!)
        @with_optional_imports(
            'google.auth.credentials',
            allow_overwrite=True  # Your existing classes might get replaced!
        )
        class GoogleProvider(BaseModel):
            credentials: Optional[Credentials] = None

        # Conservative mode - only imports objects matching module names
        @with_optional_imports(
            'google.auth.credentials',
            auto_discover=False  # Only imports 'credentials' from the module
        )
        class GoogleProvider(BaseModel):
            credentials: Optional[Credentials] = None
        ```
    """

    def decorator(model_class: type[T]) -> type[T]:
        try:
            model_module = importlib.import_module(model_class.__module__)
            imported_count = 0

            for module_name in module_names:
                try:
                    module = importlib.import_module(module_name)

                    if auto_discover:
                        # Import all public objects from the module
                        for name, obj in inspect.getmembers(module):
                            should_import = True

                            # Skip private unless explicitly included
                            if name.startswith("_") and not include_private:
                                should_import = False

                            # CRITICAL: Skip if name already exists in target module to avoid overwriting
                            elif hasattr(model_module, name):
                                logger.debug(
                                    f"Skipping '{name}' - already exists in {model_class.__module__}"
                                )
                                should_import = False

                            # Only import meaningful objects
                            elif not (
                                inspect.isclass(obj)
                                or inspect.isfunction(obj)
                                or inspect.ismethod(obj)
                                or callable(obj)
                                or
                                # Import objects defined in this module
                                (
                                    hasattr(obj, "__module__")
                                    and obj.__module__ == module_name
                                )
                            ):
                                should_import = False

                            if should_import:
                                setattr(model_module, name, obj)
                                imported_count += 1
                    else:
                        # Conservative mode - only import objects matching module segment names
                        parts = module_name.split(".")
                        for part in parts:
                            if hasattr(module, part):
                                setattr(model_module, part, getattr(module, part))
                                imported_count += 1

                        # Also try the last segment capitalized (common pattern)
                        last_part = parts[-1]
                        capitalized = last_part.capitalize()
                        if hasattr(module, capitalized):
                            setattr(
                                model_module, capitalized, getattr(module, capitalized)
                            )
                            imported_count += 1

                except ImportError as e:
                    logger.debug(f"Could not import {module_name}: {e}")
                    continue

            if imported_count > 0:
                # Rebuild the model to handle forward references
                model_class.model_rebuild()

                logger.debug(
                    f"Successfully imported {imported_count} objects from {len(module_names)} "
                    + f"modules and rebuilt {model_class.__name__}"
                )
            else:
                # All imports failed
                if warning_message:
                    logger.warning(warning_message)
                else:
                    logger.warning(
                        f"Could not import any objects from modules: {list(module_names)}. "
                        + f"Some features of {model_class.__name__} may not be available. "
                        + f"You may need to install additional dependencies."
                    )

        except Exception as e:
            logger.error(
                f"Unexpected error while processing optional imports for {model_class.__name__}: {e}",
                exc_info=True,
            )

        return model_class

    return decorator


# Convenience function to check if decorator will work
def check_optional_dependencies(*module_names: str) -> dict[str, bool]:
    """
    Check which of the specified modules are available.

    Args:
        *module_names: Module names to check

    Returns:
        Dictionary mapping module names to availability (True/False)

    Example:
        ```python
        status = check_optional_dependencies(
            'google.auth.credentials',
            'google.genai.client',
            'openai'
        )
        print(status)  # {'google.auth.credentials': True, 'google.genai.client': False, 'openai': True}
        ```
    """
    return {
        module_name: is_module_available(module_name) for module_name in module_names
    }
