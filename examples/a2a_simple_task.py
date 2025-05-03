"""
Minimal example showing how to use the A2A Interface
"""

import os
import time

from agentle.agents.a2a.a2a_interface import A2AInterface
from agentle.agents.a2a.message_parts.text_part import TextPart
from agentle.agents.a2a.messages.message import Message
from agentle.agents.a2a.tasks.task_query_params import TaskQueryParams
from agentle.agents.a2a.tasks.task_send_params import TaskSendParams
from agentle.agents.a2a.tasks.task_state import TaskState
from agentle.agents.agent import Agent
from agentle.generations.providers.google.google_genai_generation_provider import (
    GoogleGenaiGenerationProvider,
)

provider = GoogleGenaiGenerationProvider(api_key=os.environ.get("GOOGLE_API_KEY"))
agent = Agent(
    name="Example Agent", generation_provider=provider, model="gemini-2.0-flash"
)
a2a = A2AInterface(agent=agent)

# Send task to agent
message = Message(
    role="user", parts=[TextPart(text="What are three facts about the Moon?")]
)
task = a2a.tasks.send(TaskSendParams(message=message))
print(f"Task sent with ID: {task.id}")

# Wait for task completion and get result
while True:
    result = a2a.tasks.get(TaskQueryParams(id=task.id))
    status = result.result.status

    history = result.result.history
    if history is None:
        print("No history found")
        time.sleep(1)
        continue

    if status == TaskState.COMPLETED:
        print("\nResponse:", history[1].parts[0].text)
        break
    elif status == TaskState.FAILED:
        print("Task failed.")
        break
    print(f"Status: {status}")
    time.sleep(1)
