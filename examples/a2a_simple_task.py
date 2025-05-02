"""
A2A Simple Task Example

This example demonstrates how to create and monitor a task using the A2A Interface
with the InMemoryTaskManager. It shows the basic functionality of task creation
and result retrieval.
"""

import asyncio
import logging
import os
import sys
import time
import traceback
from typing import Any

from agentle.agents.a2a.a2a_interface import A2AInterface
from agentle.agents.a2a.message_parts.text_part import TextPart
from agentle.agents.a2a.messages.message import Message
from agentle.agents.a2a.models.json_rpc_response import JSONRPCResponse
from agentle.agents.a2a.resources.task_resource import TaskResource
from agentle.agents.a2a.tasks.managment.in_memory import InMemoryTaskManager
from agentle.agents.a2a.tasks.task import Task
from agentle.agents.a2a.tasks.task_get_result import TaskGetResult
from agentle.agents.a2a.tasks.task_query_params import TaskQueryParams
from agentle.agents.a2a.tasks.task_send_params import TaskSendParams
from agentle.agents.a2a.tasks.task_state import TaskState
from agentle.agents.agent import Agent
from agentle.generations.providers.google.google_genai_generation_provider import (
    GoogleGenaiGenerationProvider,
)

from dotenv import load_dotenv

load_dotenv(override=True)

# Configure root logger to see what's happening
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Add a dedicated logger for this script
logger = logging.getLogger("a2a_simple_task")


class FixedTaskResource(TaskResource):
    """
    A patched version of TaskResource that ensures proper event loop handling
    for synchronous operations.
    """

    def send(self, task: TaskSendParams) -> Task:
        """
        Sends a task to the agent with improved event loop handling.
        """
        # Create and run a new event loop for this operation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            return loop.run_until_complete(
                self.manager.send(task_params=task, agent=self.agent)
            )
        finally:
            loop.close()

    def get(self, query_params: TaskQueryParams) -> TaskGetResult:
        """
        Retrieves a task result with improved event loop handling.
        """
        # Create and run a new event loop for this operation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            return loop.run_until_complete(
                self.manager.get(query_params=query_params, agent=self.agent)
            )
        finally:
            loop.close()

    def send_subscribe(self, task: TaskSendParams) -> JSONRPCResponse:
        """
        Sends a task and subscribes to updates with improved event loop handling.
        """
        # Create and run a new event loop for this operation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            return loop.run_until_complete(
                self.manager.send_subscribe(task_params=task, agent=self.agent)
            )
        finally:
            loop.close()

    def cancel(self, task_id: str) -> bool:
        """
        Cancels an ongoing task with improved event loop handling.
        """
        # Create and run a new event loop for this operation
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            return loop.run_until_complete(self.manager.cancel(task_id=task_id))
        finally:
            loop.close()


class FixedA2AInterface(A2AInterface):
    """
    A patched version of A2AInterface that uses FixedTaskResource
    """

    def __init__(self, agent: Any, task_manager: InMemoryTaskManager):
        """Initialize with the fixed task resource"""
        super().__init__(agent=agent, task_manager=task_manager)
        # Override the tasks property with our fixed implementation
        self.tasks = FixedTaskResource(agent=agent, manager=task_manager)


def main():
    """Run the example with our simple task."""
    try:
        # Get the API key from environment
        api_key = os.environ.get("GOOGLE_API_KEY")
        logger.info(f"API key from environment: {'FOUND' if api_key else 'NOT FOUND'}")

        # Create a generation provider with explicit API key
        logger.info("Creating GoogleGenaiGenerationProvider...")
        provider = GoogleGenaiGenerationProvider(api_key=api_key)

        # Check if the provider has API key configured
        logger.info("Checking provider configuration...")
        if not provider.api_key:
            logger.warning(
                "No API key found for GoogleGenaiGenerationProvider. Please set GOOGLE_API_KEY environment variable."
            )
            print(
                "\nERROR: Google API Key is not configured. Set GOOGLE_API_KEY environment variable."
            )
            return

        # Create a simple agent
        logger.info("Creating agent...")
        agent = Agent(
            name="A2A Simple Example Agent",
            generation_provider=provider,
            model="gemini-2.0-flash",
            instructions="You are a helpful assistant. Keep your responses concise and direct.",
        )

        # Create a task manager
        logger.info("Creating task manager...")
        task_manager = InMemoryTaskManager()

        # Create the A2A interface with our fixed implementation
        logger.info("Creating Fixed A2A interface...")
        a2a_interface = FixedA2AInterface(agent=agent, task_manager=task_manager)

        # Create a simple message
        logger.info("Creating user message...")
        user_message = Message(
            role="user",
            parts=[TextPart(text="What are three interesting facts about the Moon?")],
        )

        # Create task parameters
        logger.info("Creating TaskSendParams...")
        task_params = TaskSendParams(
            message=user_message,
            sessionId="moon-facts-session",
        )

        # Send the task (using the synchronous interface)
        print("\nSending task...")
        try:
            task = a2a_interface.tasks.send(task_params)
            print(f"Task created with ID: {task.id}")
        except Exception as e:
            logger.error(f"Error sending task: {e}")
            logger.error(
                f"Traceback: {''.join(traceback.format_exception(type(e), e, e.__traceback__))}"
            )
            return

        # Poll for results with timeout
        print("\nWaiting for task to complete...")
        max_wait_time = 30  # seconds
        start_time = time.time()
        completed = False

        while time.time() - start_time < max_wait_time and not completed:
            # Check task status
            try:
                task_result = a2a_interface.tasks.get(TaskQueryParams(id=task.id))
                status = task_result.result.status
                logger.debug(
                    f"Got task with status: {status}, error: {getattr(task_result, 'error', None)}"
                )

                if status == TaskState.COMPLETED:
                    completed = True
                    print(f"\nTask completed! Final status: {status}")

                    # Extract and display the result
                    if (
                        task_result.result.history
                        and len(task_result.result.history) > 1
                    ):
                        agent_response = task_result.result.history[1]
                        if agent_response.parts:
                            print("\nAgent's response:")
                            for part in agent_response.parts:
                                if part.type == "text":
                                    print(f"{part.text}")
                elif status == TaskState.FAILED:
                    completed = True
                    print(f"\nTask failed! Error: {task_result.result.error}")
                else:
                    print(f"Current status: {status}, waiting...")
                    time.sleep(1)
            except Exception as e:
                logger.error(f"Error checking task status: {e}")
                logger.error(
                    f"Traceback: {''.join(traceback.format_exception(type(e), e, e.__traceback__))}"
                )
                time.sleep(1)

        if not completed:
            print(f"\nTask did not complete within {max_wait_time} seconds.")
            print(f"Final status: {task_result.result.status}")
            if hasattr(task_result.result, "error") and task_result.result.error:
                print(f"Error: {task_result.result.error}")

    except KeyboardInterrupt:
        print("Process interrupted by user")
    except Exception as e:
        print(f"Error occurred: {e}")
        logger.error(f"Unhandled exception: {e}")
        logger.error(
            f"Traceback: {''.join(traceback.format_exception(type(e), e, e.__traceback__))}"
        )


if __name__ == "__main__":
    main()
