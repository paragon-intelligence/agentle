from agentle.parsing.cache.in_memory_document_cache_store import (
    InMemoryDocumentCacheStore,
)
from agentle.parsing.cache.redis_cache_store import RedisCacheStore

type DocumentCacheStoreType = InMemoryDocumentCacheStore | RedisCacheStore
