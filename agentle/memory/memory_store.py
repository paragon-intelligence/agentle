import abc
from rsb.coroutines.run_sync import run_sync


class MemoryStore(abc.ABC):
    def add_memories(self, text: str, user_id: str) -> None:
        run_sync(self.add_memories_async, text=text, user_id=user_id)

    def search_memory(self, query: str, user_id: str) -> str:
        return run_sync(self.search_memory_async, query=query, user_id=user_id)

    def list_memories(self, user_id: str) -> str:
        return run_sync(self.list_memories_async, user_id=user_id)

    def delete_all_memories(self, user_id: str) -> None:
        run_sync(self.delete_all_memories_async, user_id=user_id)

    @abc.abstractmethod
    async def add_memories_async(self, text: str, user_id: str) -> None: ...

    @abc.abstractmethod
    async def search_memory_async(self, query: str, user_id: str) -> str: ...

    @abc.abstractmethod
    async def list_memories_async(self, user_id: str) -> str: ...

    @abc.abstractmethod
    async def delete_all_memories_async(self, user_id: str) -> None: ...
