from __future__ import annotations

import datetime
import uuid
from logging import Logger
from typing import TYPE_CHECKING, Literal, cast

from agentle.generations.models.generation.choice import Choice
from agentle.generations.models.generation.generation import Generation
from agentle.generations.models.generation.usage import Usage
from agentle.generations.providers.google._adapters.google_content_to_generated_assistant_message_adapter import (
    GoogleContentToGeneratedAssistantMessageAdapter,
)
from rsb.adapters.adapter import Adapter

if TYPE_CHECKING:
    from google.genai.types import Candidate, GenerateContentResponse


class GenerateGenerateContentResponseToGenerationAdapter[T](
    Adapter["GenerateContentResponse", Generation[T]]
):
    response_schema: type[T] | None
    start_time: datetime.datetime
    preferred_id: uuid.UUID | None
    model: str
    google_content_to_message_adapter: (
        GoogleContentToGeneratedAssistantMessageAdapter[T] | None
    )

    def __init__(
        self,
        *,
        model: str,
        response_schema: type[T] | None,
        start_time: datetime.datetime,
        google_content_to_generated_assistant_message_adapter: GoogleContentToGeneratedAssistantMessageAdapter[
            T
        ]
        | None = None,
        preferred_id: uuid.UUID | None = None,
    ) -> None:
        super().__init__()
        self.response_schema = response_schema
        self.start_time = start_time
        self._logger = Logger(self.__class__.__name__)
        self.google_content_to_message_adapter = (
            google_content_to_generated_assistant_message_adapter
        )
        self.preferred_id = preferred_id
        self.model = model

    def adapt(self, _f: GenerateContentResponse) -> Generation[T]:
        from google.genai import types

        parsed: T | None = cast(T | None, _f.parsed)
        candidates: list[types.Candidate] | None = _f.candidates

        if candidates is None:
            raise ValueError("The provided candidates by Google are NONE.")

        choices: list[Choice[T]] = self._build_choices(
            candidates,
            generate_content_parsed_response=parsed,
        )

        match _f.usage_metadata:
            case None:
                self._logger.warning(
                    "WARNING: No usage metadata returned by Google. Assuming 0"
                )

                usage = Usage(prompt_tokens=0, completion_tokens=0)
            case _:
                prompt_token_count = (
                    _f.usage_metadata.prompt_token_count
                    if _f.usage_metadata.prompt_token_count
                    else self._warn_and_default(field_name="prompt_token_count")
                )

                candidates_token_count = (
                    _f.usage_metadata.candidates_token_count
                    if _f.usage_metadata.candidates_token_count
                    else self._warn_and_default(field_name="candidates_token_count")
                )

                usage = Usage(
                    prompt_tokens=prompt_token_count,
                    completion_tokens=candidates_token_count,
                )

        return Generation[T](
            elapsed_time=datetime.datetime.now() - self.start_time,
            id=self.preferred_id or uuid.uuid4(),
            object="chat.generation",
            created=datetime.datetime.now(),
            model=self.model,
            choices=choices,
            usage=usage,
        )

    def _build_choices(
        self,
        candidates: list[Candidate],
        generate_content_parsed_response: T | None,
    ) -> list[Choice[T]]:
        from google.genai import types

        content_to_message_adapter = (
            self.google_content_to_message_adapter
            or GoogleContentToGeneratedAssistantMessageAdapter(
                generate_content_response_parsed=generate_content_parsed_response,
            )
        )

        choices: list[Choice[T]] = []
        index = 0
        # Build choices
        for candidate in candidates:
            candidate_content: types.Content | None = candidate.content
            if candidate_content is None:
                continue

            choices.append(
                Choice[T](
                    index=index,
                    message=content_to_message_adapter.adapt(candidate_content),
                )
            )
            index += 1

        return choices

    def _warn_and_default(self, field_name: str) -> Literal[0]:
        self._logger.warning(
            f"WARNING: No information found about {field_name}. Defaulting to 0."
        )
        return 0
