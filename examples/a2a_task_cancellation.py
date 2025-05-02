"""
A2A Task Cancellation Example

This example demonstrates how to create, monitor, and cancel tasks using the A2A Interface
with the InMemoryTaskManager. It shows how to create two tasks - one quick and one slow -
and then cancel the slow task while letting the quick task complete.
"""

import time

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


# A tool that simulates a long-running process
def slow_calculation(iterations: int):
    """
    A slow calculation that takes time to complete.
    This simulates a task that might need to be canceled.
    """
    result = {"steps": []}

    # This is intentionally slow to demonstrate cancellation
    for i in range(iterations):
        # Simulate work
        time.sleep(1)
        result["steps"].append(f"Step {i + 1} completed")

    result["final_result"] = f"Calculation completed after {iterations} iterations"
    return result


def main():
    # Create a generation provider
    provider = GoogleGenaiGenerationProvider()

    # Create an agent with a tool that takes time to complete
    agent = Agent(
        name="A2A Cancellation Example Agent",
        generation_provider=provider,
        model="gemini-2.0-flash",
        instructions="You are a helpful assistant. When asked to perform a calculation, use the slow_calculation tool.",
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
        parts=[TextPart(text="Please perform a calculation with 8 iterations.")],
    )

    # Send the quick task
    print("\nSending quick task...")
    quick_task = a2a_interface.tasks.send(
        TaskSendParams(message=quick_message, sessionId="quick-task")
    )
    print(f"Quick task created with ID: {quick_task.id}")

    # Send the slow task
    print("\nSending slow task...")
    slow_task = a2a_interface.tasks.send(
        TaskSendParams(message=slow_message, sessionId="slow-task")
    )
    print(f"Slow task created with ID: {slow_task.id}")

    # Wait briefly for tasks to start processing
    print("\nWaiting for tasks to start processing...")
    time.sleep(2)

    # Check initial status of both tasks
    print("\nInitial task status:")
    quick_result = a2a_interface.tasks.get(TaskQueryParams(id=quick_task.id))
    slow_result = a2a_interface.tasks.get(TaskQueryParams(id=slow_task.id))
    print(f"Quick task: {quick_result.result.status}")
    print(f"Slow task: {slow_result.result.status}")

    # Cancel the slow task
    print("\nCancelling slow task...")
    cancel_success = a2a_interface.tasks.cancel(slow_task.id)
    print(f"Cancel successful: {cancel_success}")

    # Wait a moment for cancellation to complete
    time.sleep(1)

    # Check status after cancellation
    print("\nStatus after cancellation:")
    quick_result = a2a_interface.tasks.get(TaskQueryParams(id=quick_task.id))
    slow_result = a2a_interface.tasks.get(TaskQueryParams(id=slow_task.id))
    print(f"Quick task: {quick_result.result.status}")
    print(f"Slow task: {slow_result.result.status}")

    # Wait for the quick task to complete
    print("\nWaiting for quick task to complete...")
    max_wait_time = 15  # seconds
    start_time = time.time()
    completed = False

    while time.time() - start_time < max_wait_time and not completed:
        quick_result = a2a_interface.tasks.get(TaskQueryParams(id=quick_task.id))
        status = quick_result.result.status

        if status == TaskState.COMPLETED:
            completed = True
            print(f"\nQuick task completed! Final status: {status}")

            # Extract and display the result
            if quick_result.result.history and len(quick_result.result.history) > 1:
                agent_response = quick_result.result.history[1]
                if agent_response.parts:
                    print("\nQuick task response:")
                    for part in agent_response.parts:
                        if part.type == "text":
                            print(f"{part.text}")
        else:
            print(f"Quick task status: {status}, waiting...")
            time.sleep(1)

    # Final status check
    print("\nFinal task status:")
    quick_result = a2a_interface.tasks.get(TaskQueryParams(id=quick_task.id))
    slow_result = a2a_interface.tasks.get(TaskQueryParams(id=slow_task.id))
    print(f"Quick task: {quick_result.result.status}")
    print(f"Slow task: {slow_result.result.status}")


if __name__ == "__main__":
    main()
