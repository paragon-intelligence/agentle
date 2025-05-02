"""
File system based prompt provider implementation.

This module provides an implementation of the PromptProvider interface that reads prompt
content from files in the local file system. It's designed for simple use cases where
prompts are stored as Markdown files in a specific directory.
"""

from pathlib import Path

from agentle.prompts.models.prompt import Prompt
from agentle.prompts.prompt_providers.prompt_provider import PromptProvider


class FSPromptProvider(PromptProvider):
    """
    A prompt provider that retrieves prompts from the file system.

    This provider reads prompt content from Markdown (.md) files located in a
    specified base directory. The prompt_id is used to construct the file path
    by appending '.md' if not already present.

    Attributes:
        base_path (str | None): The base directory path where prompt files are stored.
                               If None, relative paths will be used.
    """

    base_path: str | None

    def __init__(self, base_path: str | None = None) -> None:
        """
        Initialize a new file system prompt provider.

        Args:
            base_path (str | None, optional): The base directory path where prompt
                                             files are stored. Default is None.
        """
        super().__init__()
        self.base_path = base_path

    def provide(self, prompt_id: str, cache_ttl_seconds: int = 0) -> Prompt:
        """
        Retrieve a prompt by reading its content from a file.

        The method constructs a file path using the base_path and prompt_id,
        ensuring the file has a .md extension, then reads and returns its content
        as a Prompt object.

        Args:
            prompt_id (str): The identifier for the prompt, used to construct the file path.
            cache_ttl_seconds (int, optional): Not used in this implementation. Default is 0.

        Returns:
            Prompt: A Prompt object containing the content read from the file.

        Raises:
            FileNotFoundError: If the specified file doesn't exist.
            PermissionError: If the file cannot be read due to permissions.
            Other IO errors may also be raised.
        """
        return Prompt(
            content=Path(
                f"{self.base_path}/{prompt_id.replace('.md', '')}.md"
            ).read_text()
        )
