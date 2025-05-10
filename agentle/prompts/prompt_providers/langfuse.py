"""
Langfuse-based prompt provider implementation.

This module provides an implementation of the PromptProvider interface that fetches
prompts from the Langfuse API. It allows integration with Langfuse's prompt management
and versioning capabilities for more advanced prompt engineering workflows.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from agentle.prompts.models.prompt import Prompt
from agentle.prompts.prompt_providers.prompt_provider import PromptProvider

if TYPE_CHECKING:
    from langfuse import Langfuse


class LangfusePromptProvider(PromptProvider):
    """
    A prompt provider that retrieves prompts from the Langfuse API.

    This provider fetches prompts from Langfuse, a platform for LLM observability
    and prompt management. It requires a Langfuse client instance to be provided.

    It inherits template compilation capabilities from the base PromptProvider class,
    allowing templates to be retrieved from Langfuse and compiled with context data.

    Attributes:
        client (Langfuse): The Langfuse client instance used to connect to the API.
        cache_ttl_seconds (int): Time-to-live in seconds for caching prompts.
    """

    client: Langfuse
    cache_ttl_seconds: int = 0

    def __init__(self, client: Langfuse, cache_ttl_seconds: int = 0) -> None:
        """
        Initialize a new Langfuse prompt provider.

        Args:
            client (Langfuse): The Langfuse client instance to use for API requests.
            cache_ttl_seconds (int, optional): Time-to-live in seconds for caching.
                                              Default is 0 (no caching).
        """
        super().__init__()
        self.client = client
        self.cache_ttl_seconds = cache_ttl_seconds

    @override
    def provide(self, prompt_id: str) -> Prompt:
        """
        Retrieve a prompt from the Langfuse API.

        Fetches a prompt with the given ID from Langfuse, with optional caching.

        Args:
            prompt_id (str): The identifier for the prompt to retrieve from Langfuse.

        Returns:
            Prompt: A Prompt object containing the content fetched from Langfuse.

        Raises:
            Exceptions from the Langfuse client may be raised if the API request fails.
        """
        langfuse_prompt = self.client.get_prompt(
            prompt_id, cache_ttl_seconds=self.cache_ttl_seconds
        )

        return Prompt(content=langfuse_prompt.prompt)
