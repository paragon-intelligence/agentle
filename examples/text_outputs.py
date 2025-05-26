"""
Text Outputs Example

This example demonstrates how to create a simple agent that generates text responses
using the Agentle framework.
"""

from agentle.agents.agent import Agent
from agentle.generations.providers.google.google_genai_generation_provider import (
    GoogleGenaiGenerationProvider,
)
from agentle.generations.tracing.langfuse import LangfuseObservabilityClient

tracing_client = LangfuseObservabilityClient()

# Create a simple agent with minimal configuration
agent = Agent(
    name="Simple Text Agent",
    generation_provider=GoogleGenaiGenerationProvider(tracing_client=tracing_client),
    model="gemini-2.0-flash",  # Use an appropriate model
    instructions="You are a helpful assistant who provides concise, accurate information.",
)

# Run the agent with a simple query
response = agent.run("What are the three laws of robotics?")

# Print the response text
print(response.text)

# You can also access conversation steps - now always at least 1 step!
print(f"\nExecution steps: {len(response.context.steps)}")
for i, step in enumerate(response.context.steps):
    print(f"Step {i + 1}:")
    print(f"  Type: {step.step_type}")
    print(f"  Iteration: {step.iteration}")
    print(f"  Duration: {step.duration_ms:.1f}ms")
    print(f"  Has tool executions: {step.has_tool_executions}")
    print(f"  Successful: {step.is_successful}")
    if step.generation_text:
        preview = (
            step.generation_text[:100] + "..."
            if len(step.generation_text) > 100
            else step.generation_text
        )
        print(f"  Generated text preview: {preview}")
    if step.token_usage:
        print(
            f"  Token usage: {step.token_usage.prompt_tokens} prompt + {step.token_usage.completion_tokens} completion = {step.token_usage.total_tokens} total"
        )
