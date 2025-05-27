from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast, override

import httpx

from agentle.memory.memory_store import MemoryStore

if TYPE_CHECKING:
    from mem0.client.main import AsyncMemoryClient


class Mem0MemoryStore(MemoryStore):
    """
    Memory store that uses Mem0 as the backend.
    """

    client: AsyncMemoryClient

    def __init__(
        self,
        api_key: str | None = None,
        host: str | None = None,
        org_id: str | None = None,
        project_id: str | None = None,
        client: httpx.AsyncClient | None = None,
    ):
        from mem0.client.main import AsyncMemoryClient

        self.client = AsyncMemoryClient(
            api_key=api_key,
            host=host,
            org_id=org_id,
            project_id=project_id,
            client=client,
        )

    @override
    async def add_memories_async(self, text: str, user_id: str) -> None:
        messages = cast(list[dict[str, Any]], text)  # thanks for the poor typing, Mem0!
        memory_response = await self.client.add(
            messages=messages,
            user_id=user_id,
            metadata={"source_app": "agentle", "mcp_client": "AsyncMemoryClient"},
        )

        print(memory_response)

    @override
    async def search_memory_async(self, query: str, user_id: str) -> str: ...

    @override
    async def list_memories_async(self, user_id: str) -> str: ...

    @override
    async def delete_all_memories_async(self, user_id: str) -> None: ...


if __name__ == "__main__":
    memory_store = Mem0MemoryStore()
