"""
Tracing contracts package for Agentle framework.

This package defines interfaces and contracts for observability and tracing within the Agentle framework.
The primary class exposed is the StatefulObservabilityClient abstract base class, which defines
a contract for stateful observability clients that can track traces, generations, spans, and events
in AI applications.

Implementations of this contract (such as LangfuseObservabilityClient) allow for detailed tracing
and observability of AI model invocations, providing insights into model performance, usage patterns,
and potential issues.

The stateful design of these clients enables method chaining and hierarchical tracing structures,
where traces can contain spans, which can contain generations, etc.

Example:
```python
from agentle.generations.tracing.contracts import StatefulObservabilityClient
from agentle.generations.tracing.langfuse import LangfuseObservabilityClient

# Create a client
client = LangfuseObservabilityClient()

# Use method chaining to create a trace with nested spans and generations
response = (
    client.trace(name="user_request", user_id="user123")
    .span(name="process_query", input={"query": "What's the weather in New York?"})
    .generation(name="weather_model", input={"location": "New York"})
    .end(output={"forecast": "Sunny, 25°C"})
    .end()  # End the span
    .end(output={"response": "The weather in New York is sunny with 25°C"})  # End the trace
)
```
"""

from agentle.generations.tracing.contracts.stateful_observability_client import (
    StatefulObservabilityClient,
)

__all__ = ["StatefulObservabilityClient"]
