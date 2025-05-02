"""
A2A Interface Example

This example demonstrates how to use the A2A Interface to interact with an agent
using a standardized protocol interface.
"""

from agentle.agents.a2a.tasks.task_query_params import TaskQueryParams
from agentle.agents.agent import Agent
from agentle.agents.a2a.a2a_interface import A2AInterface
from agentle.agents.a2a.tasks.managment.task_manager import TaskManager
from agentle.agents.a2a.tasks.task_send_params import TaskSendParams
from agentle.agents.a2a.messages.message import Message
from agentle.agents.a2a.message_parts.text_part import TextPart
from agentle.generations.providers.google.google_genai_generation_provider import (
    GoogleGenaiGenerationProvider,
)

# Create a generation provider
provider = GoogleGenaiGenerationProvider()

# Create a simple agent
agent = Agent(
    name="A2A Example Agent",
    generation_provider=provider,
    model="gemini-2.0-flash",
    instructions="""You are a helpful assistant that provides clear and concise responses.
    Focus on giving accurate information in a friendly tone.
    """,
)

# Create a task manager (typically this would be part of a server application)
task_manager = TaskManager()

# Create the A2A interface with our agent
a2a_interface = A2AInterface(agent=agent, task_manager=task_manager)

# Create a message for the agent
user_message = Message(
    role="user",
    parts=[
        TextPart(text="What are three interesting facts about artificial intelligence?")
    ],
)

# Create a task send parameters object
task_params = TaskSendParams(
    message=user_message,
    sessionId="example-session-1",
)

print("Sending task to the agent...")

# Use the A2A interface to send the task
task = a2a_interface.tasks.send(task_params)

print(f"Task created with ID: {task.id}")
print(f"Task status: {task.status}")

# Get the task result
task_result = a2a_interface.tasks.get(TaskQueryParams(id=task.id))

print("\nTask Result:")
print("-" * 80)

# The result contains the agent's response
if task_result.result.history and len(task_result.result.history) > 1:
    agent_response = task_result.result.history[
        1
    ]  # The agent's response comes after the user's message
    if agent_response.parts:
        for part in agent_response.parts:
            if part.type == "text":
                print(part.text)
print("-" * 80)

# You can also create multiple tasks or continue an existing conversation
# by using the same sessionId
follow_up_message = Message(
    role="user",
    parts=[TextPart(text="Explain more about the second fact you mentioned.")],
)

follow_up_params = TaskSendParams(
    message=follow_up_message,
    sessionId="example-session-1",  # Same session ID to continue the conversation
)

print("\nSending follow-up task...")
follow_up_task = a2a_interface.tasks.send(follow_up_params)

follow_up_result = a2a_interface.tasks.get(query_params={"id": follow_up_task.id})

print("\nFollow-up Result:")
print("-" * 80)
if follow_up_result.result.history and len(follow_up_result.result.history) > 3:
    agent_response = follow_up_result.result.history[3]  # The agent's new response
    if agent_response.parts:
        for part in agent_response.parts:
            if part.type == "text":
                print(part.text)
print("-" * 80)

# The A2A interface provides a standardized way to interact with agents
# that can be used across different agent implementations (Agent, AgentTeam, AgentPipeline)
