import abc


class MemoryStore(abc.ABC):
    @abc.abstractmethod
    async def add_memories_async(self, text: str, user) -> None: ...

    @abc.abstractmethod
    async def search_memory_async(self, query: str) -> str: ...

    @abc.abstractmethod
    async def list_memories_async(self) -> str: ...

    @abc.abstractmethod
    async def delete_all_memories_async(self) -> None: ...
