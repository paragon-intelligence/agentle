"""
A2A Simple Task Example (Synchronous Version)

This example demonstrates how to create and monitor tasks using the A2A Interface
with the InMemoryTaskManager, using a fully synchronous approach.
"""

import time
from typing import Dict, List

from agentle.agents.a2a.a2a_interface import A2AInterface
from agentle.agents.a2a.message_parts.text_part import TextPart
from agentle.agents.a2a.messages.message import Message
from agentle.agents.a2a.tasks.managment.in_memory import InMemoryTaskManager
from agentle.agents.a2a.tasks.task_query_params import TaskQueryParams
from agentle.agents.a2a.tasks.task_send_params import TaskSendParams
from agentle.agents.a2a.tasks.task_state import TaskState
from agentle.agents.agent import Agent
from agentle.generations.providers.google.google_genai_generation_provider import (
    GoogleGenaiGenerationProvider,
)


# Define a simple tool to list items
def list_items() -> Dict[str, List[str]]:
    """
    A simple tool that returns a list of sample items.
    """
    return {
        "items": [
            "Laptop",
            "Smartphone",
            "Headphones",
            "Coffee mug",
            "Notebook",
        ]
    }


# Create a generation provider
provider = GoogleGenaiGenerationProvider()

# Create an agent with a simple tool
agent = Agent(
    name="A2A Simple Example Agent",
    generation_provider=provider,
    model="gemini-2.0-flash",
    instructions="""You are a helpful assistant that can list items.
    When asked to list items, use the list_items tool.
    """,
    tools=[list_items],
)

# Create a task manager
task_manager = InMemoryTaskManager()

# Create the A2A interface
a2a_interface = A2AInterface(agent=agent, task_manager=task_manager)

# Create messages for different tasks
list_items_message = Message(
    role="user",
    parts=[TextPart(text="Please list the available items.")],
)

greeting_message = Message(
    role="user",
    parts=[TextPart(text="Hello, how are you today?")],
)

# Create task parameters
list_items_task_params = TaskSendParams(
    message=list_items_message,
    sessionId="list-items-session",
)

greeting_task_params = TaskSendParams(
    message=greeting_message,
    sessionId="greeting-session",
)

# Send the first task
print("\nSending task to list items...")
list_items_task = a2a_interface.tasks.send(list_items_task_params)
print(f"List items task created with ID: {list_items_task.id}")

# Send the second task
print("\nSending greeting task...")
greeting_task = a2a_interface.tasks.send(greeting_task_params)
print(f"Greeting task created with ID: {greeting_task.id}")


# Function to check and print task status
def check_task_status(task_id, task_name):
    result = a2a_interface.tasks.get(TaskQueryParams(id=task_id))
    status = result.result.status
    print(f"{task_name} status: {status}")

    if (
        status == TaskState.COMPLETED
        and result.result.history
        and len(result.result.history) > 1
    ):
        agent_response = result.result.history[1]
        if agent_response.parts:
            for part in agent_response.parts:
                if part.type == "text":
                    print(f"{task_name} response: {part.text}")

    return status


# Wait for tasks to complete, with a timeout
print("\nWaiting for tasks to complete...")
timeout = 30  # seconds
start_time = time.time()

while time.time() - start_time < timeout:
    time.sleep(2)

    list_items_status = check_task_status(list_items_task.id, "List items task")
    greeting_status = check_task_status(greeting_task.id, "Greeting task")

    print("")  # Empty line for better readability

    # Check if both tasks are completed or failed
    if list_items_status in [
        TaskState.COMPLETED,
        TaskState.FAILED,
    ] and greeting_status in [TaskState.COMPLETED, TaskState.FAILED]:
        break

print("\nTask processing complete!")

# Print final results
print("\nFinal Results:")
check_task_status(list_items_task.id, "List items task")
check_task_status(greeting_task.id, "Greeting task")
