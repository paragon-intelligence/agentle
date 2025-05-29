from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import override
from uuid import UUID

from agentle.rag.documents.models.blob import Blob
from agentle.rag.documents.models.document import Document
from agentle.rag.documents.repositories.downloadable import AsyncDownloadable
from rsb.contracts.repositories.deletable import AsyncDeletable
from rsb.contracts.repositories.readable import (
    AsyncBulkReader,
    AsyncReader,
)
from rsb.contracts.repositories.writable import AsyncWritable


class NullDocumentRepository(
    AsyncReader[Document],
    AsyncBulkReader[Document],
    AsyncWritable[Path],
    AsyncDeletable[str],
    AsyncDownloadable[Blob],
):
    """
    Implementação do padrão Null Object para DocumentRepository.
    Esta classe implementa todas as interfaces necessárias, mas não executa nenhuma operação real.
    Útil para testes, desenvolvimento e quando um repositório real não está disponível.
    """

    @override
    async def read_async(
        self, uid: str, filters: dict[str, object] | None = None
    ) -> Document:
        """Retorna um documento vazio."""
        return Document(id=UUID(uid), chunks=[], metadata={})

    @override
    async def read_all_async(
        self, filters: dict[str, object] | None = None
    ) -> Sequence[Document]:
        """Retorna uma lista vazia de documentos."""
        return []

    @override
    async def delete_async(self, uid: str) -> None:
        """Não executa nenhuma operação de exclusão."""
        pass

    @override
    async def download_async(self, uid: str) -> Blob:
        """Retorna um blob vazio."""
        return Blob(data=b"", extension="")

    @override
    async def write_async(self, e: Path) -> None:
        """Não executa nenhuma operação de escrita."""
        pass
