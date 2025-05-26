"""
Defines the Prompt model for representing and manipulating prompt content.

This module provides the core Prompt class which represents a text prompt that can
contain variable placeholders, conditional blocks, and iteration blocks. It supports
advanced template compilation with a Handlebars-like syntax.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any, TypeVar, cast

from rsb.decorators.value_objects import valueobject
from rsb.models.base_model import BaseModel

T = TypeVar("T")


@valueobject
class Prompt(BaseModel):
    """
    Represents a text prompt that can contain template expressions.

    A Prompt instance manages a text content that can include various template
    features like variable placeholders ({{variable_name}}), conditional blocks
    ({{#if condition}}...{{/if}}), and iteration blocks ({{#each items}}...{{/each}}).

    These template expressions can be processed using the compile method.

    Attributes:
        content (str): The text content of the prompt.
        compiled (bool): Flag indicating if the prompt has been compiled
                        (had its template expressions processed). Default is False.
    """

    content: str
    compiled: bool = False

    def compile(
        self, context: dict[str, Any] | None = None, **replacements: Any
    ) -> Prompt:
        """
        Create a new Prompt with template expressions processed.

        This method supports a Handlebars-like templating system with:
        - Variable interpolation: {{variable_name}}
        - Conditional blocks: {{#if condition}}...{{/if}}
        - Iteration blocks: {{#each items}}...{{/each}}
        - Nested property access: {{object.property}}

        The method can be used in two ways:
        1. With a context dictionary: prompt.compile({"name": "World"})
        2. With keyword arguments: prompt.compile(name="World")

        Args:
            context (dict[str, Any], optional): A dictionary with values for template variables.
            **replacements: Keyword arguments with values for template variables.
                            Only used if context is None.

        Returns:
            Prompt: A new Prompt instance with the template expressions processed

        Raises:
            ValueError: If the template syntax is invalid

        Examples:
            Simple variable replacement:
            >>> prompt = Prompt("Hello, {{name}}!")
            >>> prompt.compile(name="World")
            Prompt(content="Hello, World!")

            Using conditional blocks:
            >>> prompt = Prompt("{{#if show_greeting}}Hello{{/if}}, {{name}}!")
            >>> prompt.compile(show_greeting=True, name="World")
            Prompt(content="Hello, World!")

            Using iteration:
            >>> prompt = Prompt("Items: {{#each items}}- {{this.name}}\\n{{/each}}")
            >>> prompt.compile(items=[{"name": "Apple"}, {"name": "Banana"}])
            Prompt(content="Items: - Apple\\n- Banana\\n")
        """
        # If context is not provided, use the replacements
        if context is None:
            context = replacements

        # For simple cases without advanced template features, use fast path
        if not any(marker in self.content for marker in ["{{#if", "{{#each"]):
            return self._simple_compile(context)

        # For advanced template features, use the full template processing
        result = self.content

        # Process conditional blocks {{#if condition}}...{{/if}}
        result = self._process_conditionals(result, context)

        # Process iteration blocks {{#each items}}...{{/each}}
        result = self._process_iterations(result, context)

        # Process simple variable interpolation {{variable}}
        result = self._process_variables(result, context)

        return Prompt(content=result, compiled=True)

    def _simple_compile(self, context: dict[str, Any]) -> Prompt:
        """
        Perform simple variable replacement without processing control structures.
        This is faster for templates that only use variable interpolation.
        """
        content = self.content

        def replace_var(match: re.Match[str]) -> str:
            var_name = match.group(1).strip()
            value = self._get_nested_value(var_name, context)
            return str(value) if value is not None else match.group(0)

        # Replace all {{variable}} patterns
        pattern = r"{{([^#/][^}]*?)}}"
        content = re.sub(pattern, replace_var, content)

        return Prompt(content=content, compiled=True)

    def _process_conditionals(self, template: str, context: dict[str, Any]) -> str:
        """Process all conditional blocks in the template."""
        # Find all conditional blocks
        pattern = r"{{#if\s+([^}]+)}}(.*?){{/if}}"

        # Keep processing until all conditionals are resolved
        while re.search(pattern, template, re.DOTALL):

            def evaluate_match(match: re.Match[str]) -> str:
                return self._evaluate_conditional(match, context)

            template = re.sub(
                pattern,
                evaluate_match,
                template,
                flags=re.DOTALL,
            )

        return template

    def _evaluate_conditional(
        self, match: re.Match[str], context: dict[str, Any]
    ) -> str:
        """Evaluate a single conditional block."""
        condition_var = match.group(1).strip()
        content = match.group(2)

        # Check if condition exists in context
        value = self._get_nested_value(condition_var, context)

        # If condition is truthy, return content, otherwise empty string
        if value:
            return content
        return ""

    def _process_iterations(self, template: str, context: dict[str, Any]) -> str:
        """Process all iteration blocks in the template."""
        # Find all iteration blocks
        pattern = r"{{#each\s+([^}]+)}}(.*?){{/each}}"

        # Keep processing until all iterations are resolved
        while re.search(pattern, template, re.DOTALL):

            def evaluate_match(match: re.Match[str]) -> str:
                return self._evaluate_iteration(match, context)

            template = re.sub(
                pattern,
                evaluate_match,
                template,
                flags=re.DOTALL,
            )

        return template

    def _evaluate_iteration(self, match: re.Match[str], context: dict[str, Any]) -> str:
        """Evaluate a single iteration block."""
        items_var = match.group(1).strip()
        item_template = match.group(2)

        # Get the iterable from context
        items = self._get_nested_value(items_var, context)

        if not items or not isinstance(items, (list, tuple, dict)):
            return ""  # Return empty if not iterable

        result: list[str] = []

        # Handle the iteration
        iterable = cast(Iterable[Any], items)
        for item in iterable:
            # Create a temporary context with 'this' referring to the current item
            temp_context = dict(context)
            temp_context["this"] = item

            # Replace variables in the item template
            item_result = self._process_variables(item_template, temp_context)
            result.append(item_result)

        return "".join(result)

    def _process_variables(self, template: str, context: dict[str, Any]) -> str:
        """Replace all variable placeholders with their values."""

        def replace_var(match: re.Match[str]) -> str:
            var_name = match.group(1).strip()

            # Get the value from context, supporting nested properties
            value = self._get_nested_value(var_name, context)

            # Convert value to string or empty string if None
            return str(value) if value is not None else ""

        # Replace all {{variable}} patterns
        pattern = r"{{([^#/][^}]*?)}}"
        return re.sub(pattern, replace_var, template)

    def _get_nested_value(self, path: str, context: dict[str, Any]) -> Any:
        """
        Get a value from the context dict using dot notation for nested properties.

        Args:
            path (str): The path to the value (e.g., "user.address.city")
            context (dict[str, Any]): The context dictionary

        Returns:
            Any: The value at the specified path or None if not found
        """
        current = context

        # Handle the special case of 'this'
        if path == "this":
            return current.get("this")

        # Split path by dots and traverse the context
        keys = path.split(".")

        for key in keys:
            key = key.strip()

            if isinstance(current, dict) and key in current:
                current = cast(dict[str, Any], current[key])
            elif hasattr(cast(dict[str, Any], current), key):
                # Support for object attributes
                current = getattr(current, key)  # type: ignore
            else:
                return None

        return current

    @property
    def text(self) -> str:
        """
        Get the content of the prompt as a string.

        Returns:
            str: The content of the prompt.
        """
        return self.content

    def __str__(self) -> str:
        """
        String representation of the prompt.

        Returns:
            str: The content of the prompt.
        """
        return self.text

    def __add__(self, other: Prompt | str) -> Prompt:
        """
        Concatenate two prompts or a prompt and a string.
        """
        if isinstance(other, str):
            return Prompt(content=self.text + other)
        return Prompt(content=self.text + other.text)

    def __getitem__(self, key: int | slice) -> str:
        """
        Support indexing and slicing operations on the prompt content.

        Args:
            key: Integer index or slice object

        Returns:
            str: Character at index or substring for slice

        Examples:
            >>> prompt = Prompt("Hello World")
            >>> prompt[0]  # 'H'
            >>> prompt[6:]  # 'World'
            >>> prompt[:5]  # 'Hello'
        """
        return self.content[key]
