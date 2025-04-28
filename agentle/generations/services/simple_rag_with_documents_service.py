from __future__ import annotations

from collections.abc import Sequence
from typing import override

from agentle.rag.documents.models.document import Document
from agentle.generations.collections.message_sequence import MessageSequence
from agentle.generations.models.messages.developer_message import DeveloperMessage
from agentle.generations.models.generation.generation import Generation
from agentle.generations.models.messages.message import Message
from agentle.generations.models.structured_outputs_store.query_expansion import (
    QueryExpansion,
)
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.providers import GenerationProvider
from agentle.generations.services.generations_service_protocol import (
    GenerationsServiceProtocol,
)
from agentle.generations.tracing.contracts.stateful_observability_client import (
    StatefulObservabilityClient,
)
from rsb.contracts.maybe_protocol import MaybeProtocol
from rsb.contracts.repositories.readable import AsyncBulkReader
from rsb.containers.maybe import Maybe

type WithoutStructuredOutput = None


class SimpleRAGWithDocumentsService(GenerationsServiceProtocol):
    documents_repository: AsyncBulkReader[Document] | None
    observability_client: MaybeProtocol[StatefulObservabilityClient]

    def __init__(
        self,
        generation_strategy: GenerationProvider,
        documents_repository: AsyncBulkReader[Document] | None = None,
        observability_client: StatefulObservabilityClient | None = None,
    ) -> None:
        super().__init__(generation_strategy)
        self.documents_repository = documents_repository
        self.observability_client = Maybe(observability_client)

    @override
    async def generate_async[T = WithoutStructuredOutput](
        self,
        model: str,
        messages: Sequence[Message],
        response_schema: type[T] | None = None,
    ) -> Generation[T]:
        if not self.documents_repository:
            return await self.generation_provider.create_generation_async(
                model=model,
                messages=messages,
                response_schema=response_schema,
            )

        related_documents_instructions: list[Message] = []

        message_sequence = MessageSequence(messages)

        no_developer_prompt_messages = message_sequence.without_developer_prompt()

        query_expansion_message = DeveloperMessage(
            parts=[
                TextPart(
                    text=(
                        "You are an AI agent that analyses a conversation, analyses if it's necessary"
                        + "to generate a query to access additional information to provide the answer to the user "
                        + "question. If it's necessary, write a good query to access the database. If it's not, "
                        + "just return a null query."
                    )
                )
            ]
        )

        expanded_query_generation = (
            await self.generation_provider.create_generation_async(
                model=model,
                messages=[query_expansion_message] + list(no_developer_prompt_messages),
                response_schema=QueryExpansion,
            )
        )

        expanded_query = expanded_query_generation.parsed.expanded_query
        if expanded_query:
            related_documents = await self.documents_repository.read_all(
                filters={"query": expanded_query}
            )

            related_documents_instructions.append(
                UserMessage(
                    parts=[
                        TextPart(
                            text="".join(
                                document.describe() for document in related_documents
                            )
                        )
                    ]
                )
            )

        return await self.generation_provider.create_generation_async(
            model=model,
            messages=related_documents_instructions,
            response_schema=response_schema,
        )
