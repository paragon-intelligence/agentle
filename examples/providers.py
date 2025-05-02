"""
Providers Example

This example demonstrates how to use different model providers with the Agentle framework.
"""

from agentle.agents.agent import Agent
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.providers.base.generation_provider import GenerationProvider
from agentle.generations.providers.google.google_genai_generation_provider import (
    GoogleGenaiGenerationProvider,
)

# Example 1: Create an agent with Google's Gemini model
provider: GenerationProvider = GoogleGenaiGenerationProvider()

# Run the Google agent
google_response = provider.create_generation(
    messages=[
        UserMessage(
            content="Explain the concept of neural networks briefly.",
        )
    ]
)
print("GOOGLE GEMINI RESPONSE:")
print(google_response.text)
print("\n" + "-" * 50 + "\n")


# Example 2: Creating an agent with a custom provider
# Note: This is a placeholder. In a real application, you would implement your own provider.
class CustomProviderExample:
    """
    This is a placeholder for demonstrating how you would implement a custom provider.
    In a real application, you would create a class that implements the GenerationProvider interface.
    """

    def __init__(self):
        print("In a real application, this would initialize your custom provider.")
        print("You would need to implement methods like:")
        print("- create_generation_async")
        print("- create_chat_generation_async")
        print("See the GoogleGenaiGenerationProvider for an example implementation.")


print("CUSTOM PROVIDER EXAMPLE:")
custom_provider = CustomProviderExample()
print("\n" + "-" * 50 + "\n")


# Example 3: Switching between providers with the same agent
print("SWITCHING PROVIDERS EXAMPLE:")
print(
    "You can easily switch providers by creating a new agent or using the clone method:"
)

original_agent = Agent(
    name="Original Agent",
    generation_provider=GoogleGenaiGenerationProvider(),
    model="gemini-2.0-flash",
    instructions="You are a helpful assistant.",
)

# Clone the agent with a hypothetical different provider
cloned_agent_explanation = """
# Clone with a different provider (example code):
different_provider = DifferentProvider()

new_agent = original_agent.clone(
    new_name="New Provider Agent",
    new_generation_provider=different_provider,
    new_model="different-model"
)
"""

print(cloned_agent_explanation)
