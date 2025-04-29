"""
Configuration options for function calling behavior with Google AI models.

This module defines the FunctionCallingConfig TypedDict, which controls how function
calling is handled when interacting with Google's Generative AI models. These settings
allow fine-tuning of the function calling behavior to meet specific application needs.

The configuration options include:
- disable: Whether to disable function calling entirely
- maximum_remote_calls: Maximum number of function calls allowed per request
- ignore_call_history: Whether to ignore previous function call results in the context
"""

from typing import NotRequired, TypedDict


class FunctionCallingConfig(TypedDict):
    """
    Configuration options for controlling function calling behavior with Google AI models.

    Attributes:
        disable: Whether to disable automatic function calling. Defaults to True if not specified.
        maximum_remote_calls: Maximum number of remote function calls allowed in a single request.
        ignore_call_history: Whether to ignore previous function call history in the current context.
    """

    disable: NotRequired[bool]
    maximum_remote_calls: NotRequired[int]
    ignore_call_history: NotRequired[bool]
