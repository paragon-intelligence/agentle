"""
Adapter module for converting Google AI GenerateContentResponse objects to Agentle Generation objects.

This module provides the GenerateGenerateContentResponseToGenerationAdapter class, which
transforms responses from Google's Generative AI API into the standardized Generation
format used throughout the Agentle framework. The adapter handles conversion of response
content, metadata, and usage statistics.

This adapter is a critical component of Agentle's provider abstraction layer, enabling
the framework to present a unified interface regardless of which underlying AI provider
is being used. It processes all the provider-specific details of Google's response format
and normalizes them to Agentle's internal representation.

The adapter supports structured output parsing for type-safe responses through its
generic type parameter.

Example:
```python
from agentle.generations.providers.google._adapters.generate_generate_content_response_to_generation_adapter import (
    GenerateGenerateContentResponseToGenerationAdapter
)
import datetime
from pydantic import BaseModel

# Define a structured response type
class WeatherInfo(BaseModel):
    location: str
    temperature: float
    conditions: str

# Create the adapter
start_time = datetime.datetime.now()
adapter = GenerateGenerateContentResponseToGenerationAdapter[WeatherInfo](
    model="gemini-1.5-pro",
    response_schema=WeatherInfo,
    start_time=start_time
)

# When a response is received from Google's API
generation = adapter.adapt(google_response)

# Access the standardized result
print(f"Model: {generation.model}")
print(f"Completion: {generation.text}")
print(f"Tokens used: {generation.usage.total_tokens}")

# Access structured data
if generation.parsed:
    print(f"Weather in {generation.parsed.location}: {generation.parsed.temperature}°C, {generation.parsed.conditions}")
```
"""

from __future__ import annotations

import datetime
import uuid
from logging import Logger
from typing import TYPE_CHECKING, Literal, cast

from rsb.adapters.adapter import Adapter

from agentle.generations.models.generation.choice import Choice
from agentle.generations.models.generation.generation import Generation
from agentle.generations.models.generation.usage import Usage
from agentle.generations.providers.google.adapters.google_content_to_generated_assistant_message_adapter import (
    GoogleContentToGeneratedAssistantMessageAdapter,
)

if TYPE_CHECKING:
    from google.genai.types import Candidate, GenerateContentResponse


class GenerateGenerateContentResponseToGenerationAdapter[T](
    Adapter["GenerateContentResponse", Generation[T]]
):
    """
    Adapter for converting Google AI GenerateContentResponse objects to Agentle Generation objects.

    This adapter transforms the response format from Google's Generative AI API
    into Agentle's standardized Generation format. It processes elements such as
    candidate responses, usage statistics, and structured output data.

    The adapter is generic over type T, which represents the optional structured
    data format that can be extracted from the model's response when a response
    schema is provided.

    This class plays a key role in Agentle's provider abstraction layer by normalizing
    Google-specific response formats into the framework's unified representation.

    Attributes:
        response_schema (type[T] | None): The optional Pydantic model class used to
            parse structured data from the response. When provided, the adapter will
            attempt to extract typed data according to this schema.

        preferred_id (uuid.UUID | None): An optional UUID to use for the resulting
            Generation object. If not provided, a new UUID is generated.

        model (str): The name of the model that was used to generate the response.
            This value is included in the resulting Generation object.

        google_content_to_message_adapter (GoogleContentToGeneratedAssistantMessageAdapter[T] | None):
            An optional adapter for converting Google Content objects to Agentle message
            objects. If not provided, a new adapter will be created as needed.

    Example:
        ```python
        # Create an adapter for a simple text completion
        adapter = GenerateGenerateContentResponseToGenerationAdapter(
            model="gemini-1.5-pro",
            response_schema=None,  # No structured data
        )

        # Process a response from Google's API
        agentle_generation = adapter.adapt(google_response)

        # For structured data parsing
        from pydantic import BaseModel

        class MovieInfo(BaseModel):
            title: str
            director: str
            year: int

        structured_adapter = GenerateGenerateContentResponseToGenerationAdapter[MovieInfo](
            model="gemini-1.5-pro",
            response_schema=MovieInfo,
        )

        # When Google returns parsed data according to the schema
        movie_generation = structured_adapter.adapt(parsed_google_response)

        if movie_generation.parsed:
            print(f"{movie_generation.parsed.title} ({movie_generation.parsed.year})")
            print(f"Directed by: {movie_generation.parsed.director}")
        ```
    """

    response_schema: type[T] | None
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
        google_content_to_generated_assistant_message_adapter: GoogleContentToGeneratedAssistantMessageAdapter[
            T
        ]
        | None = None,
        preferred_id: uuid.UUID | None = None,
    ) -> None:
        """
        Initialize the adapter with the necessary configuration.

        Args:
            model: The name of the model that generated the response.
            response_schema: Optional Pydantic model class used to parse structured
                data from the response.
            start_time: The timestamp when the generation request was initiated.
            google_content_to_generated_assistant_message_adapter: Optional adapter for
                converting Google Content objects to Agentle message objects.
            preferred_id: Optional UUID to use for the resulting Generation object.
        """
        super().__init__()
        self.response_schema = response_schema
        self._logger = Logger(self.__class__.__name__)
        self.google_content_to_message_adapter = (
            google_content_to_generated_assistant_message_adapter
        )
        self.preferred_id = preferred_id
        self.model = model

    def adapt(self, _f: GenerateContentResponse) -> Generation[T]:
        """
        Convert a Google GenerateContentResponse to an Agentle Generation object.

        This method processes the response from Google's API, extracting candidates,
        usage statistics, and any structured data. It then constructs a standardized
        Generation object that can be used consistently throughout the Agentle framework.

        Args:
            _f: The Google GenerateContentResponse object to adapt, typically received
                directly from Google's Generative AI API.

        Returns:
            Generation[T]: An Agentle Generation object containing the normalized response
                data, including any parsed structured output if a response_schema was provided.

        Raises:
            ValueError: If the provided Google response doesn't contain any candidates.

        Note:
            If usage statistics are not available in the Google response, the adapter
            will log a warning and default to zero values to ensure a valid Generation
            object is still created.
        """
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
        """
        Build Choice objects from Google candidate responses.

        This internal method processes each candidate in the Google response
        and converts it to Agentle's Choice format. It handles adaptation of
        the content to Agentle message format and indexing of multiple choices.

        Args:
            candidates: List of candidate responses from Google's API.
            generate_content_parsed_response: Optional parsed structured data
                extracted from the model response.

        Returns:
            list[Choice[T]]: A list of Choice objects representing each candidate
                response from the model, adapted to Agentle's format.
        """
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
        """
        Log a warning about missing metadata and return a default value.

        This internal utility method logs a warning when expected usage metadata
        is not available in the Google response and returns a default value of 0.

        Args:
            field_name: The name of the missing metadata field, used in the warning message.

        Returns:
            Literal[0]: Always returns 0 as the default value for missing fields.
        """
        self._logger.warning(
            f"WARNING: No information found about {field_name}. Defaulting to 0."
        )
        return 0
