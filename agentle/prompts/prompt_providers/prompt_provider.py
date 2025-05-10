"""
Defines the abstract base class for prompt providers.

This module establishes the common interface that all prompt providers must implement,
enabling a consistent way to retrieve prompts regardless of their source.
"""

from __future__ import annotations

import abc

from agentle.prompts.models.prompt import Prompt


class PromptProvider(abc.ABC):
    """
    Abstract base class defining the interface for all prompt providers.

    A PromptProvider is responsible for retrieving prompts from a specific source.
    Different implementations can fetch prompts from various backends like
    file systems, APIs, databases, etc.

    All concrete implementations must override the provide method.
    """

    @abc.abstractmethod
    def provide(self, prompt_id: str) -> Prompt:
        """
        Retrieve a prompt by its identifier.

        Args:
            prompt_id (str): The unique identifier for the prompt to retrieve.
            cache_ttl_seconds (int, optional): Time-to-live in seconds for caching
                                              the prompt. Default is 0 (no caching).

        Returns:
            Prompt: The retrieved prompt.

        Raises:
            Implementation-specific exceptions may be raised if the prompt
            cannot be retrieved.
        """
        ...
