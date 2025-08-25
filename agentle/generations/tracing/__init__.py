"""
Simplified tracing module for the Agentle framework.

This module provides a high-performance, minimal tracing solution that integrates
with Langfuse V3. The architecture consists of three main components:

1. TracingClient - Simple base interface
2. LangfuseTracingClient - High-performance Langfuse V3 implementation
3. @observe decorator - Zero-overhead tracing decorator

Usage:
```python
from agentle.generations.tracing.xxxxx import LangfuseTracingClient, observe

# Set up tracing client
tracing_client = LangfuseTracingClient()

# Apply to your provider
class MyProvider(GenerationProviderType):
    def __init__(self):
        super().__init__(tracing_client=tracing_client)

    @observe
    async def generate_async(self, ...):
        # Your generation logic
        return generation
```

The design prioritizes performance through:
- Fire-and-forget tracing operations
- Early exit when tracing is disabled
- Minimal async/await usage
- Simple, flat API without complex hierarchies
"""
