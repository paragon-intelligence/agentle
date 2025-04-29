"""
Generations package for Agentle framework.

This package provides the core functionality for generating content using various AI models
and providers. It includes tools for model integration, parameter configuration, tool calling,
and observability.

Key components of this package include:

- Tools: Functions that can be called by AI models to perform specific actions
- Tracing: Observability and monitoring for AI model invocations
- Models: Data models for representing generations and related entities
- Providers: Implementations for various AI model providers (OpenAI, Google, etc.)
- Collections: Utilities for working with collections of generations
- JSON: Utilities for JSON serialization and deserialization
- Pricing: Utilities for calculating and tracking costs

Example:
```python
from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider
from agentle.generations.models.generation.generation_config import GenerationConfig
from agentle.generations.models.message import UserMessage
from agentle.generations.models.message_parts.text import TextPart

# Create a provider for generating content
provider = GoogleGenerationProvider()

# Generate content using the provider
generation = provider.create_generation(
    model="gemini-1.5-flash",
    messages=[
        UserMessage(parts=[
            TextPart(text="What are three interesting facts about Tokyo?")
        ])
    ],
    generation_config=GenerationConfig(temperature=0.7)
)

# Use the generated content
print(generation.text)
```
"""
