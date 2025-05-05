"""
Text Outputs Example

This example demonstrates how to create a simple agent that generates text responses
using the Agentle framework.
"""

from agentle.agents.agent import Agent
from agentle.generations.providers.google.google_genai_generation_provider import (
    GoogleGenaiGenerationProvider,
)

# Create a simple agent with minimal configuration
agent = Agent(
    name="Simple Text Agent",
    generation_provider=GoogleGenaiGenerationProvider(),
    model="gemini-2.0-flash",  # Use an appropriate model
    instructions="You are a helpful assistant who provides concise, accurate information.",
)

# Run the agent with a simple query
response = agent.run("What are the three laws of robotics?")

# Print the response text
print(response.text)

# You can also access conversation steps if needed
print("\nConversation steps:")
for i, step in enumerate(response.steps):
    print(f"Step {i + 1}: {step}")

# To run asynchronously:
# import asyncio
# async def main():
#     response = await agent.run_async("What are the three laws of robotics?")
#     print(response.text)
# asyncio.run(main())
