from __future__ import annotations

from pathlib import Path
from typing import override

from agentle.rag.documents.services.ingestion.document_ingestion_service_protocol import (
    DocumentIngestionServiceProtocol,
)


class DefaultDocumentIngestionService(DocumentIngestionServiceProtocol):
    @override
    async def create_document_async(self, file_path: str) -> None:
        await self.writable_repository.write_async(Path(file_path))
