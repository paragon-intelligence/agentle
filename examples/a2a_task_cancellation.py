"""
A2A Task Cancellation Example (Synchronous Version)

This example demonstrates how to create, monitor, and cancel tasks using the A2A Interface
with the InMemoryTaskManager, using a fully synchronous approach.
"""

import time
from typing import Any, Dict

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


# Create a tool that simulates a long-running process
def slow_calculation(iterations: int) -> Dict[str, Any]:
    """
    A slow calculation that takes time to complete.
    This simulates a long-running task that might need to be canceled.
    """
    result = {"steps": []}

    # This is a slow calculation that we want to be able to cancel
    for i in range(iterations):
        # Simulate work
        time.sleep(1)
        result["steps"].append(f"Step {i + 1} completed")

    result["final_result"] = f"Calculation completed after {iterations} iterations"
    return result


# Create a generation provider
provider = GoogleGenaiGenerationProvider()

# Create an agent with a tool that takes time to complete
agent = Agent(
    name="A2A Cancellation Example Agent",
    generation_provider=provider,
    model="gemini-2.0-flash",
    instructions="""You are a helpful assistant that can perform calculations.
    When asked to perform a calculation, use the slow_calculation tool.
    """,
    tools=[slow_calculation],
)

# Create a task manager
task_manager = InMemoryTaskManager()

# Create the A2A interface
a2a_interface = A2AInterface(agent=agent, task_manager=task_manager)

# Create messages for different tasks
quick_message = Message(
    role="user",
    parts=[TextPart(text="What is the capital of France?")],
)

slow_message = Message(
    role="user",
    parts=[TextPart(text="Please perform a calculation with 10 iterations.")],
)

# Create task parameters
quick_task_params = TaskSendParams(
    message=quick_message,
    sessionId="quick-session",
)

slow_task_params = TaskSendParams(
    message=slow_message,
    sessionId="slow-session",
)

# Send the quick task
print("\nSending quick task...")
quick_task = a2a_interface.tasks.send(quick_task_params)
print(f"Quick task created with ID: {quick_task.id}")

# Send the slow task
print("\nSending slow task...")
slow_task = a2a_interface.tasks.send(slow_task_params)
print(f"Slow task created with ID: {slow_task.id}")

# Wait a moment for tasks to start processing
print("\nWaiting for tasks to start processing...")
time.sleep(2)

# Check status of both tasks
quick_result = a2a_interface.tasks.get(TaskQueryParams(id=quick_task.id))
slow_result = a2a_interface.tasks.get(TaskQueryParams(id=slow_task.id))

print(f"\nQuick task status: {quick_result.result.status}")
print(f"Slow task status: {slow_result.result.status}")

# Cancel the slow task
print("\nCancelling slow task...")
cancel_success = a2a_interface.tasks.cancel(slow_task.id)
print(f"Cancel successful: {cancel_success}")

# Wait a moment for cancellation to complete
print("\nWaiting for cancellation to complete...")
time.sleep(2)

# Check status after cancellation
quick_result = a2a_interface.tasks.get(TaskQueryParams(id=quick_task.id))
slow_result = a2a_interface.tasks.get(TaskQueryParams(id=slow_task.id))

print(f"\nQuick task status after cancellation: {quick_result.result.status}")
print(f"Slow task status after cancellation: {slow_result.result.status}")

# Wait for the quick task to complete if it hasn't already
if quick_result.result.status not in [
    TaskState.COMPLETED,
    TaskState.FAILED,
    TaskState.CANCELED,
]:
    print("\nWaiting for quick task to complete...")
    while True:
        time.sleep(1)
        quick_result = a2a_interface.tasks.get(TaskQueryParams(id=quick_task.id))
        if quick_result.result.status in [
            TaskState.COMPLETED,
            TaskState.FAILED,
            TaskState.CANCELED,
        ]:
            break

# Get final results
final_quick_result = a2a_interface.tasks.get(TaskQueryParams(id=quick_task.id))
final_slow_result = a2a_interface.tasks.get(TaskQueryParams(id=slow_task.id))

print("\nFinal Results:")
print(f"Quick task final status: {final_quick_result.result.status}")
print(f"Slow task final status: {final_slow_result.result.status}")

if final_quick_result.result.history and len(final_quick_result.result.history) > 1:
    agent_response = final_quick_result.result.history[1]
    if agent_response.parts:
        for part in agent_response.parts:
            if part.type == "text":
                print(f"\nQuick task response: {part.text}")
