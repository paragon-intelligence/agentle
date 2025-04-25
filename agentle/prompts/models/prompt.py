from __future__ import annotations

from rsb.decorators.value_objects import valueobject
from rsb.models.base_model import BaseModel


@valueobject
class Prompt(BaseModel):
    content: str
    compiled: bool = False

    def compile(
        self, **replacements: Prompt | str | int | float | bool | None
    ) -> Prompt:
        """
        Create a new Prompt with variables in the content replaced with provided values.

        Variables in the content are denoted by double curly braces: {{variable_name}}.
        Each occurrence of {{variable_name}} will be replaced with the string representation
        of the corresponding value in the replacements dictionary.

        If a variable in the content doesn't have a corresponding replacement,
        it remains unchanged in the output.

        Args:
            **replacements: Dictionary of variable names to replacement values.
                        Values will be converted to strings using str().
                        Common types include: str, int, float, bool, None.

        Returns:
            Prompt: A new Prompt instance with the replacements applied.

        Examples:
            >>> prompt = Prompt("Hello, {{name}}!")
            >>> prompt.compile(name="World")
            Prompt(content="Hello, World!")

            >>> prompt = Prompt("The answer is {{number}}.")
            >>> prompt.compile(number=42)
            Prompt(content="The answer is 42.")
        """
        content = self.content
        for key, value in replacements.items():
            placeholder = f"{{{{{key}}}}}"
            content = content.replace(placeholder, str(value))

        return Prompt(content=str(content), compiled=True)

    @property
    def text(self) -> str:
        return self.content

    def __str__(self) -> str:
        return self.text
