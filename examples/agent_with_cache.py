"""
Example demonstrating cache store usage with agents.

This example shows how to use different cache stores (InMemory and Redis)
for caching parsed documents to improve performance.
"""

import asyncio
from agentle.agents.agent import Agent
from agentle.agents.knowledge.static_knowledge import StaticKnowledge
from agentle.generations.providers.google.google_generation_provider import (
    GoogleGenerationProvider,
)
from agentle.parsing.cache import InMemoryDocumentCacheStore, RedisCacheStore


async def main():
    """Demonstrate cache store usage with agents."""

    # Example 1: Using InMemory cache store (default)
    print("=== Example 1: InMemory Cache Store ===")

    in_memory_cache = InMemoryDocumentCacheStore(
        cleanup_interval=60
    )  # Cleanup every minute

    agent_with_memory_cache = Agent(
        name="Research Assistant with Memory Cache",
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.0-flash",
        instructions="You are a research assistant that analyzes documents.",
        static_knowledge=[
            # Cache for 1 hour
            StaticKnowledge(content="https://example.com/research.pdf", cache=3600),
            # Cache indefinitely
            StaticKnowledge(content="local_document.txt", cache="infinite"),
        ],
        document_cache_store=in_memory_cache,
        debug=True,
    )

    # First run - will parse and cache documents
    print("First run (cache miss expected):")
    response1 = await agent_with_memory_cache.run_async("What are the key findings?")
    print(
        f"Response: {response1.generation.text[:100] if response1.generation else 'No response'}..."
    )

    # Get cache statistics
    stats = in_memory_cache.get_stats()
    print(f"Cache stats: {stats}")

    # Second run - should use cached documents
    print("\nSecond run (cache hit expected):")
    response2 = await agent_with_memory_cache.run_async("Summarize the methodology.")
    print(
        f"Response: {response2.generation.text[:100] if response2.generation else 'No response'}..."
    )

    # Example 2: Using Redis cache store (for distributed environments)
    print("\n=== Example 2: Redis Cache Store ===")

    try:
        redis_cache = RedisCacheStore(
            redis_url="redis://localhost:6379/0",
            key_prefix="agentle:demo:",
            default_ttl=7200,  # 2 hours default
        )

        agent_with_redis_cache = Agent(
            name="Research Assistant with Redis Cache",
            generation_provider=GoogleGenerationProvider(),
            model="gemini-2.0-flash",
            instructions="You are a research assistant that analyzes documents.",
            static_knowledge=[
                # Cache for 30 minutes
                StaticKnowledge(content="https://example.com/data.pdf", cache=1800),
                # Cache for 1 day
                StaticKnowledge(content="important_document.docx", cache=86400),
            ],
            document_cache_store=redis_cache,
            debug=True,
        )

        # Run with Redis cache
        print("Running with Redis cache:")
        response3 = await agent_with_redis_cache.run_async("What are the conclusions?")
        print(
            f"Response: {response3.generation.text[:100] if response3.generation else 'No response'}..."
        )

        # Get Redis cache info
        cache_info = await redis_cache.get_cache_info()
        print(f"Redis cache info: {cache_info}")

        # Clean up Redis connection
        await redis_cache.close()

    except ImportError:
        print("Redis not available. Install with: pip install redis")
    except Exception as e:
        print(f"Redis connection failed: {e}")

    # Example 3: No cache (default behavior)
    print("\n=== Example 3: No Cache (Default) ===")

    agent_no_cache = Agent(
        name="Research Assistant without Cache",
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.0-flash",
        instructions="You are a research assistant that analyzes documents.",
        static_knowledge=[
            # No caching (cache=None is default)
            "This is raw text knowledge that won't be cached.",
            StaticKnowledge(
                content="https://example.com/report.pdf"
            ),  # No cache specified
        ],
        # No document_cache_store specified - will use default InMemoryDocumentCacheStore
        debug=True,
    )

    print("Running without explicit caching:")
    response4 = await agent_no_cache.run_async("What are the recommendations?")
    print(
        f"Response: {response4.generation.text[:100] if response4.generation else 'No response'}..."
    )

    # Example 4: Cache management operations
    print("\n=== Example 4: Cache Management ===")

    cache = InMemoryDocumentCacheStore()

    # Check if a specific document is cached
    cache_key = cache.get_cache_key("example_document.pdf", "PDFParser")
    exists = await cache.exists_async(cache_key)
    print(f"Document cached: {exists}")

    # Clear all cache entries
    await cache.clear_async()
    print("Cache cleared")

    # Get updated stats
    final_stats = cache.get_stats()
    print(f"Final cache stats: {final_stats}")


if __name__ == "__main__":
    asyncio.run(main())
