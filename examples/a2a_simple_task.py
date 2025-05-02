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


async def main_async():
    """Run the example with our simple task asynchronously."""
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

        # Create the A2A interface
        logger.info("Creating A2A interface...")
        a2a_interface = A2AInterface(agent=agent, task_manager=task_manager)

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
            agent=agent,  # Explicitly pass the agent to the task
        )

        # Send the task (using async version directly)
        print("\nSending task...")
        try:
            task = await a2a_interface.task_manager.send(task_params, agent=agent)
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
                task_result = await a2a_interface.task_manager.get(
                    TaskQueryParams(id=task.id), agent=agent
                )
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
                    await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error checking task status: {e}")
                logger.error(
                    f"Traceback: {''.join(traceback.format_exception(type(e), e, e.__traceback__))}"
                )
                await asyncio.sleep(1)

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


def main():
    """Run the async main function using asyncio.run."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
