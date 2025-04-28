from __future__ import annotations

import copy
import logging
import uuid
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Literal, overload

from pydantic import BaseModel
from rsb.decorators.entities import entity

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
    elapsed_time: timedelta
    id: uuid.UUID
    object: Literal["chat.generation"]
    created: datetime
    model: str
    choices: Sequence[Choice[T]]
    usage: Usage

    @property
    def parsed(self) -> T:
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
        if len(self.choices) > 1:
            logger.warning(
                "WARNING: choices list is > 1. Coudn't determine the parts. Returning the first choice parts."
            )

        return self.get_message_parts(0)

    def get_message_parts(
        self, choice: int
    ) -> Sequence[TextPart | ToolExecutionSuggestion]:
        return self.choices[choice].message.parts

    @property
    def tool_calls(self) -> Sequence[ToolExecutionSuggestion]:
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
                elapsed_time=new_elapsed_time or self.elapsed_time,
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
                elapsed_time=new_elapsed_time or self.elapsed_time,
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
                elapsed_time=new_elapsed_time or self.elapsed_time,
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
        return len(self.tool_calls)

    def get_tool_calls(self, choice: int = 0) -> Sequence[ToolExecutionSuggestion]:
        return self.choices[choice].message.tool_calls

    @classmethod
    def mock(cls) -> Generation[T]:
        return cls(
            model="mock-model",
            elapsed_time=timedelta(seconds=0),
            id=uuid.uuid4(),
            object="chat.generation",
            created=datetime.now(),
            choices=[],
            usage=Usage(prompt_tokens=0, completion_tokens=0),
        )

    @property
    def text(self) -> str:
        return "".join([choice.message.text for choice in self.choices])

    def get_parsed(self, choice: int) -> T:
        return self.choices[choice].message.parsed
