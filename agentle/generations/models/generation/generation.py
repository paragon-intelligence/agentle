"""
Core Generation model for AI responses in the Agentle framework.

This module defines the Generation class, which serves as the core container for
AI-generated content in the Agentle framework. It encapsulates all aspects of a
model's response, including the generated content itself, metadata, usage statistics,
and structured data parsing.

The Generation model is designed to be provider-independent, allowing applications
to work with different AI providers through a unified interface. It supports
multiple response choices, structured data parsing via generic typing, and
provides convenient accessors for common operations.

The model includes extensive clone functionality to support transformation and
manipulation of generation objects while maintaining immutability.
"""

from __future__ import annotations

import copy
import logging
import uuid
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Literal, overload

from rsb.decorators.entities import entity
from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.generations.models.generation.choice import Choice
from agentle.generations.models.generation.usage import Usage
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.message_parts.tool_execution_suggestion import (
    ToolExecutionSuggestion,
)
from agentle.generations.models.messages.generated_assistant_message import (
    GeneratedAssistantMessage,
)

logger = logging.getLogger(__name__)


@entity
class Generation[T](BaseModel):
    """
    Primary container for AI-generated content with metadata.

    The Generation class encapsulates a complete response from an AI model,
    including the generated content, metadata about the generation process,
    usage statistics, and potential structured data of type T.

    This class serves as the central return type for all provider implementations,
    ensuring a consistent interface regardless of which AI provider is being used.
    It supports multiple response choices (alternatives), type-safe structured data
    access, and convenient accessors for commonly needed information.

    The generic type parameter T allows for structured output parsing, enabling
    type-safe access to parsed data when a response_schema is provided.

    Attributes:
        elapsed_time: Time taken to generate the response
        id: Unique identifier for this generation
        object: Type identifier, always "chat.generation"
        created: Timestamp when this generation was created
        model: Identifier of the model that produced this generation
        choices: Sequence of alternative responses from the model
        usage: Token usage statistics for this generation
    """

    id: uuid.UUID = Field(
        description="Unique identifier for tracking and referencing this specific generation throughout the system. Used for logging, debugging, and associating generations with specific requests.",
        examples=[uuid.uuid4(), uuid.uuid4(), uuid.uuid4()],
    )
    object: Literal["chat.generation"] = Field(
        description="Type discriminator that identifies this object as a generation. Always set to 'chat.generation' to support polymorphic handling of different response types.",
        examples=["chat.generation"],
    )
    created: datetime = Field(
        description="ISO 8601 timestamp when this generation was created. Useful for tracking generation history, calculating processing time, and implementing time-based features.",
        examples=[datetime.now(), datetime.now() - timedelta(minutes=5)],
    )
    model: str = Field(
        description="Identifier string for the AI model that produced this generation. Includes provider and model name/version information to enable model-specific handling and analytics.",
        examples=["gpt-4-turbo", "claude-3-sonnet", "llama-3-70b-instruct"],
    )
    choices: Sequence[Choice[T]] = Field(
        description="Collection of alternative responses from the model when multiple completions are requested. Each choice contains a generated message with text content, tool calls, and optional parsed structured data.",
    )
    usage: Usage = Field(
        description="Token usage statistics for tracking resource consumption and cost. Contains counts for tokens in the prompt and completion, enabling precise usage tracking and cost estimation across providers.",
        examples=[
            Usage(prompt_tokens=150, completion_tokens=50),
            Usage(prompt_tokens=800, completion_tokens=200),
        ],
    )

    @property
    def parsed(self) -> T:
        """
        Get the parsed structured data from the first choice.

        This is a convenience property that returns the parsed data from the
        first choice in the choices sequence. It's useful when you only have
        one choice and want direct access to the parsed data.

        Returns:
            T: The parsed structured data from the first choice

        Raises:
            ValueError: If there are multiple choices, as it's ambiguous
                which one to use
        """
        if len(self.choices) > 1:
            raise ValueError(
                "Choices list is > 1. Coudn't determine the parsed "
                + "model to obtain. Please, use the get_parsed "
                + "method, instead, passing the choice number "
                + "you want to get the parsed model."
            )

        return self.get_parsed(0)

    @property
    def parts(self) -> Sequence[TextPart | ToolExecutionSuggestion]:
        """
        Get the message parts from the first choice.

        This is a convenience property that returns the message parts from the
        first choice in the choices sequence. It includes both text parts and
        tool execution suggestions.

        Returns:
            Sequence[TextPart | ToolExecutionSuggestion]: The message parts
                from the first choice
        """
        if len(self.choices) > 1:
            logger.warning(
                "WARNING: choices list is > 1. Coudn't determine the parts. Returning the first choice parts."
            )

        return self.get_message_parts(0)

    def get_message_parts(
        self, choice: int
    ) -> Sequence[TextPart | ToolExecutionSuggestion]:
        """
        Get the message parts from a specific choice.

        Args:
            choice: The index of the choice to get message parts from

        Returns:
            Sequence[TextPart | ToolExecutionSuggestion]: The message parts
                from the specified choice
        """
        return self.choices[choice].message.parts

    @property
    def tool_calls(self) -> Sequence[ToolExecutionSuggestion]:
        """
        Get tool execution suggestions from the first choice.

        This is a convenience property that returns only the tool execution
        suggestions from the first choice in the choices sequence.

        Returns:
            Sequence[ToolExecutionSuggestion]: The tool execution suggestions
                from the first choice
        """
        if len(self.choices) > 1:
            logger.warning(
                "Choices list is > 1. Coudn't determine the tool calls. "
                + "Please, use the get_tool_calls method, instead, "
                + "passing the choice number you want to get the tool calls."
                + "Returning the first choice tool calls."
            )

        return self.get_tool_calls(0)

    @overload
    def clone[T_Schema](
        self,
        *,
        new_parseds: Sequence[T_Schema],
        new_elapsed_time: timedelta | None = None,
        new_id: uuid.UUID | None = None,
        new_object: Literal["chat.generation"] | None = None,
        new_created: datetime | None = None,
        new_model: str | None = None,
        new_choices: None = None,
        new_usage: Usage | None = None,
    ) -> Generation[T_Schema]: ...

    @overload
    def clone[T_Schema](
        self,
        *,
        new_parseds: None = None,
        new_elapsed_time: timedelta | None = None,
        new_id: uuid.UUID | None = None,
        new_object: Literal["chat.generation"] | None = None,
        new_created: datetime | None = None,
        new_model: str | None = None,
        new_choices: Sequence[Choice[T_Schema]],
        new_usage: Usage | None = None,
    ) -> Generation[T_Schema]: ...

    @overload
    def clone(
        self,
        *,
        # Nenhum destes é fornecido para este overload
        new_parseds: None = None,
        new_choices: None = None,
        # Apenas estes podem ser fornecidos
        new_elapsed_time: timedelta | None = None,
        new_id: uuid.UUID | None = None,
        new_object: Literal["chat.generation"] | None = None,
        new_created: datetime | None = None,
        new_model: str | None = None,
        new_usage: Usage | None = None,
    ) -> Generation[T]: ...  # Retorna o mesmo tipo T

    def clone[T_Schema](  # type: ignore[override]
        self,
        *,
        new_parseds: Sequence[T_Schema] | None = None,
        new_elapsed_time: timedelta | None = None,
        new_id: uuid.UUID | None = None,
        new_object: Literal["chat.generation"] | None = None,
        new_created: datetime | None = None,
        new_model: str | None = None,
        new_choices: Sequence[Choice[T_Schema]] | None = None,
        new_usage: Usage | None = None,
    ) -> Generation[T_Schema] | Generation[T]:  # Adjusted return type hint for clarity
        """
        Create a clone of this Generation, optionally with modified attributes.

        This method creates a new Generation object based on the current one,
        with the option to modify specific attributes. It supports several scenarios:

        1. Creating a new Generation with the same structure but different parsed data
        2. Creating a new Generation with entirely new choices
        3. Creating a simple clone with optional metadata changes

        The method uses overloads to provide proper type safety depending on which
        scenario is being used.

        Args:
            new_parseds: New parsed data to use in place of existing parsed data
            new_elapsed_time: New elapsed time value
            new_id: New ID for the generation
            new_object: New object type identifier
            new_created: New creation timestamp
            new_model: New model identifier
            new_choices: New choices to replace the existing ones
            new_usage: New usage statistics

        Returns:
            A new Generation object with the requested modifications

        Raises:
            ValueError: If both new_parseds and new_choices are provided, which
                would be ambiguous
        """
        # Validate against ambiguous parameter usage
        if new_choices and new_parseds:
            raise ValueError(
                "Cannot provide 'new_choices' together with 'new_parseds'."
            )

        # Scenario 1: Clone with new parsed data
        if new_parseds:
            # Validate length consistency
            if len(new_parseds) != len(self.choices):
                raise ValueError(
                    f"The number of 'new_parseds' ({len(new_parseds)}) does not match the number of existing 'choices' ({len(self.choices)})."
                )

            _new_choices_scenario1: list[Choice[T_Schema]] = [
                Choice(
                    message=GeneratedAssistantMessage(
                        # Use deepcopy for parts to ensure independence
                        parts=copy.deepcopy(choice.message.parts),
                        parsed=new_parseds[choice.index],
                    ),
                    index=choice.index,
                )
                for choice in self.choices
            ]

            return Generation[T_Schema](
                id=new_id or self.id,
                object=new_object or self.object,
                created=new_created or self.created,
                model=new_model or self.model,
                choices=_new_choices_scenario1,
                usage=(new_usage or self.usage).model_copy(),
            )

        # Scenario 2: Clone with entirely new choices provided
        if new_choices:
            return Generation[T_Schema](
                id=new_id or self.id,
                object=new_object or self.object,
                created=new_created or self.created,
                model=new_model or self.model,
                choices=new_choices,
                usage=(new_usage or self.usage).model_copy(),
            )

        # Scenario 3: Simple clone (same type T), potentially updating metadata
        if not new_parseds and not new_choices:
            # Deep copy existing choices to ensure independence
            _new_choices_scenario3: list[Choice[T]] = [
                copy.deepcopy(choice) for choice in self.choices
            ]
            # Cast is needed because the method signature expects T_Schema, but in this branch,
            # we know we are returning Generation[T]. Overloads handle the public API typing.
            return Generation[T](  # type: ignore[return-value]
                id=new_id or self.id,
                object=new_object or self.object,
                created=new_created or self.created,
                model=new_model or self.model,
                choices=_new_choices_scenario3,  # type: ignore[arg-type]
                usage=(new_usage or self.usage).model_copy(),
            )

        # Should be unreachable if overloads cover all valid cases and validation works
        raise ValueError(
            "Invalid combination of parameters for clone method. Use one of the defined overloads."
        )

    def tool_calls_amount(self) -> int:
        """
        Get the number of tool execution suggestions in the first choice.

        Returns:
            int: The number of tool execution suggestions
        """
        return len(self.tool_calls)

    def get_tool_calls(self, choice: int = 0) -> Sequence[ToolExecutionSuggestion]:
        """
        Get tool execution suggestions from a specific choice.

        Args:
            choice: The index of the choice to get tool calls from (default: 0)

        Returns:
            Sequence[ToolExecutionSuggestion]: The tool execution suggestions
                from the specified choice
        """
        return self.choices[choice].message.tool_calls

    @classmethod
    def mock(cls) -> Generation[T]:
        """
        Create a mock Generation object for testing purposes.

        This method creates a Generation with minimal default values,
        useful for testing without making actual API calls.

        Returns:
            Generation[T]: A mock Generation object
        """
        return cls(
            model="mock-model",
            id=uuid.uuid4(),
            object="chat.generation",
            created=datetime.now(),
            choices=[],
            usage=Usage(prompt_tokens=0, completion_tokens=0),
        )

    @property
    def text(self) -> str:
        """
        Get the concatenated text from all choices.

        This is a convenience property that returns all the text content
        from all choices concatenated into a single string.

        Returns:
            str: The concatenated text from all choices
        """
        return "".join([choice.message.text for choice in self.choices])

    def get_parsed(self, choice: int) -> T:
        """
        Get the parsed structured data from a specific choice.

        Args:
            choice: The index of the choice to get parsed data from

        Returns:
            T: The parsed structured data from the specified choice
        """
        return self.choices[choice].message.parsed
