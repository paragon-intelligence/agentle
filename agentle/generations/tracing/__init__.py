"""
Tracing package for Agentle framework.

This package provides observability and tracing capabilities for the Agentle framework,
allowing for detailed tracking and monitoring of AI model invocations, agent activities,
and system interactions.

The package includes both abstract interfaces (contracts) that define the expected behavior
of observability clients, as well as concrete implementations such as the Langfuse-based client.

Tracing in Agentle follows a hierarchical model where:
- Traces represent end-to-end user interactions or system processes
- Spans represent subtasks or phases within a trace
- Generations represent individual AI model invocations
- Events represent discrete points of interest within a process

All of these can be recorded, tracked, and analyzed to understand system behavior,
identify performance bottlenecks, and debug issues.

Example:
```python
from agentle.generations.tracing import LangfuseObservabilityClient

# Create a tracing client
tracer = LangfuseObservabilityClient()

# Start a trace for a user interaction
with_trace = tracer.trace(
    name="process_user_query",
    user_id="user123",
    input={"query": "What's the weather in Tokyo?"}
)

# Record a model generation within the trace
with_generation = with_trace.generation(
    name="weather_prediction",
    input={"location": "Tokyo"},
    metadata={"model": "weather-model-v2"}
)

# Complete the generation with its output
with_generation.end(output={"forecast": "Sunny, 28°C"})

# Complete the trace with the final response
with_trace.end(output={"response": "The weather in Tokyo is sunny with 28°C"})
```
"""

from agentle.generations.tracing.contracts import StatefulObservabilityClient
from agentle.generations.tracing.langfuse import LangfuseObservabilityClient

__all__ = ["StatefulObservabilityClient", "LangfuseObservabilityClient"]
