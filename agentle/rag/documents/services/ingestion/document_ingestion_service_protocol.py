from __future__ import annotations

from abc import abstractmethod
from pathlib import Path

from rsb.contracts.repositories.writable import AsyncWritable
from rsb.coroutines.run_sync import run_sync
from rsb.decorators.services import abstractservice


@abstractservice
class DocumentIngestionServiceProtocol:
    writable_repository: AsyncWritable[Path]

    def __init__(self, writable_repository: AsyncWritable[Path]) -> None:
        self.writable_repository = writable_repository

    def create_document(self, file_path: str) -> None:
        return run_sync(self.create_document_async, timeout=None, file_path=file_path)

    @abstractmethod
    async def create_document_async(self, file_path: str) -> None: ...
