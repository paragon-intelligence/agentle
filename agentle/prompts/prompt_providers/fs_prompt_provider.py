"""
File system based prompt provider implementation.

This module provides an implementation of the PromptProvider interface that reads prompt
content from files in the local file system with thread-safe TTL-based caching.
"""

from __future__ import annotations
import time
import threading
import weakref
from pathlib import Path
from typing import Literal, override

from agentle.prompts.models.prompt import Prompt
from agentle.prompts.prompt_providers.prompt_provider import PromptProvider

type Deactivated = None


class FSPromptProvider(PromptProvider):
    """
    A prompt provider that retrieves prompts from the file system with TTL-based caching.

    This provider reads prompt content from Markdown (.md), text (.txt), or XML (.xml)
    files located in a specified base directory. The prompt_id is used to construct the file path.

    The caching system is thread-safe and does not block the main thread. Cached entries
    automatically expire after their TTL without requiring manual intervention.

    Attributes:
        base_path (str | None): The base directory path where prompt files are stored.
                               If None, relative paths will be used.
        cache (Literal["infinite"] | int | None):
            - "infinite": Cache entries never expire
            - int value: Cache entries expire after this many seconds
            - None: No caching is performed
    """

    base_path: str | None
    cache: Literal["infinite"] | int | Deactivated
    _cache_store: dict[str, tuple[Prompt, float]]  # {prompt_id: (prompt, timestamp)}
    _cache_lock: threading.RLock
    _cleanup_timer: threading.Timer | None
    _cleanup_interval: int

    # Class-level tracking for resource management
    _instances: weakref.WeakSet[FSPromptProvider] = weakref.WeakSet()
    _class_lock = threading.RLock()

    def __init__(
        self,
        base_path: str | None = None,
        cache: Literal["infinite"] | int | Deactivated | None = None,
        cleanup_interval: int = 60,  # Run cleanup every 60 seconds by default
    ) -> None:
        """
        Initialize a new file system prompt provider.

        Args:
            base_path (str | None, optional): The base directory path where prompt
                                             files are stored. Default is None.
            cache (Literal["infinite"] | int | None, optional):
                Configure caching behavior:
                - "infinite": Cache entries never expire
                - int value: Cache entries expire after this many seconds
                - None: No caching is performed
            cleanup_interval (int, optional): How often to run the cache cleanup
                                             task in seconds. Default is 60.
        """
        super().__init__()
        self.base_path = base_path
        self.cache = cache
        self._cache_store = {}
        self._cache_lock = threading.RLock()
        self._cleanup_interval = cleanup_interval
        self._cleanup_timer = None

        # Register this instance
        with self._class_lock:
            self._instances.add(self)

        # Start the cleanup timer if needed
        if self.cache is not None and self.cache != "infinite":
            self._start_cleanup_timer()

    @override
    def provide(self, prompt_id: str) -> Prompt:
        """
        Retrieve a prompt by reading its content from a file.

        This method is thread-safe and works in both synchronous and
        asynchronous environments.

        Args:
            prompt_id (str): The identifier for the prompt.

        Returns:
            Prompt: A Prompt object containing the content read from the file.

        Raises:
            FileNotFoundError: If no matching file exists.
            PermissionError: If the file cannot be read due to permissions.
            Other IO errors may also be raised.
        """
        # Check cache first if enabled
        if self._is_cache_valid(prompt_id):
            with self._cache_lock:
                # Return a copy to avoid potential concurrent modification issues
                prompt, _ = self._cache_store[prompt_id]
                return prompt

        # Get the file path
        file_path = self._get_file_path(prompt_id)

        # Read the prompt content
        prompt_content = file_path.read_text(encoding="utf-8")
        prompt = Prompt(content=prompt_content)

        # Store in cache if caching is enabled
        if self.cache is not None:
            with self._cache_lock:
                self._cache_store[prompt_id] = (prompt, time.time())

        return prompt

    def _start_cleanup_timer(self) -> None:
        """Start a timer that periodically cleans up expired cache entries."""
        with self._cache_lock:
            # Cancel any existing timer
            if self._cleanup_timer is not None:
                self._cleanup_timer.cancel()

            # Create a new timer
            self._cleanup_timer = threading.Timer(
                self._cleanup_interval, self._timer_callback
            )
            self._cleanup_timer.daemon = True  # Don't keep the application running
            self._cleanup_timer.start()

    def _timer_callback(self) -> None:
        """Callback function for the timer to clean cache and restart timer."""
        try:
            self._cleanup_expired_cache()
        except Exception as e:
            # Log the error but don't crash
            print(f"Error during cache cleanup: {e}")
        finally:
            # Restart the timer if this instance still exists
            if self in self._instances:
                self._start_cleanup_timer()

    def _cleanup_expired_cache(self) -> None:
        """
        Clean up expired cache entries in a thread-safe manner.
        """
        if self.cache is None or self.cache == "infinite":
            return

        current_time = time.time()
        with self._cache_lock:
            expired_keys = [
                key
                for key, (_, timestamp) in self._cache_store.items()
                if current_time - timestamp >= self.cache
            ]

            for key in expired_keys:
                del self._cache_store[key]

    def __del__(self) -> None:
        """Clean up resources when the object is garbage collected."""
        # Stop the cleanup timer if it's running
        if hasattr(self, "_cleanup_timer") and self._cleanup_timer is not None:
            self._cleanup_timer.cancel()
            self._cleanup_timer = None

        # Remove this instance from the set of instances
        with self._class_lock:
            if hasattr(self, "_instances") and self in self._instances:
                self._instances.remove(self)

    def _get_file_path(self, prompt_id: str) -> Path:
        """
        Get the file path for a prompt ID.

        This method tries .md, .txt, and .xml extensions if not already specified.
        If prompt_id contains forward slashes (e.g., 'folder/subfolder/name'),
        it will be treated as a path relative to base_dir.

        Args:
            prompt_id (str): The prompt identifier, which can include directory paths

        Returns:
            Path: Path object for the prompt file

        Raises:
            FileNotFoundError: If no matching file is found
        """
        base_dir = Path(self.base_path) if self.base_path else Path()

        # Check if prompt_id already has an extension
        if prompt_id.endswith((".md", ".txt", ".xml")):
            file_path = base_dir / prompt_id
            if file_path.exists():
                return file_path

        # Try extensions in order of preference: .md, .xml, .txt
        for extension in [".md", ".xml", ".txt"]:
            file_path = base_dir / f"{prompt_id}{extension}"
            if file_path.exists():
                return file_path

        # If we got here, no matching file was found
        raise FileNotFoundError(f"No prompt file found for ID: {prompt_id}")

    def _is_cache_valid(self, prompt_id: str) -> bool:
        """
        Check if the cached prompt is still valid.

        Args:
            prompt_id (str): The prompt identifier

        Returns:
            bool: True if cache is valid, False otherwise
        """
        with self._cache_lock:
            if self.cache is None or prompt_id not in self._cache_store:
                return False

            if self.cache == "infinite":
                return True

            # Check if cache has expired
            _, timestamp = self._cache_store[prompt_id]
            is_valid = (
                time.time() - timestamp < self.cache
            )  # cache is valid if within TTL

            # If expired, remove from cache
            if not is_valid:
                del self._cache_store[prompt_id]

            return is_valid
