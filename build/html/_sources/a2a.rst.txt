===
A2A
===

Agentle provides built-in support for the Agent-to-Agent (A2A) Protocol, enabling standardized communication between agents. This page explains how to use the A2A features in Agentle.

What is A2A?
-----------

A2A (Agent-to-Agent) is an open protocol designed to enable standardized communication between autonomous agents built on different frameworks and by various vendors. It provides a common interface for agent communication, task management, and state tracking.

Basic A2A Usage
-------------

Here's a simple example of using the A2A interface:

.. code-block:: python

    import os
    import time

    from agentle.agents.a2a.a2a_interface import A2AInterface
    from agentle.agents.a2a.message_parts.text_part import TextPart
    from agentle.agents.a2a.messages.message import Message
    from agentle.agents.a2a.tasks.task_query_params import TaskQueryParams
    from agentle.agents.a2a.tasks.task_send_params import TaskSendParams
    from agentle.agents.a2a.tasks.task_state import TaskState
    from agentle.agents.agent import Agent
    from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider

    # Set up agent and A2A interface
    provider = GoogleGenerationProvider(api_key=os.environ.get("GOOGLE_API_KEY"))
    agent = Agent(
        name="Example Agent", 
        generation_provider=provider, 
        model="gemini-2.5-flash"
    )
    a2a = A2AInterface(agent=agent)

    # Send task to agent
    message = Message(
        role="user", 
        parts=[TextPart(text="What are three facts about the Moon?")]
    )
    task = a2a.tasks.send(TaskSendParams(message=message))
    print(f"Task sent with ID: {task.id}")

    # Wait for task completion and get result
    while True:
        result = a2a.tasks.get(TaskQueryParams(id=task.id))
        status = result.result.status
        
        if status == TaskState.COMPLETED:
            print("\nResponse:", result.result.history[1].parts[0].text)
            break
        elif status == TaskState.FAILED:
            print(f"Task failed: {result.result.error}")
            break
        print(f"Status: {status}")
        time.sleep(1)

A2A Components
------------

The A2A implementation in Agentle consists of several key components:

A2AInterface
~~~~~~~~~~~

The main entry point for A2A functionality:

.. code-block:: python

    from agentle.agents.a2a.a2a_interface import A2AInterface
    from agentle.agents.agent import Agent

    # Create an agent
    agent = Agent(
        name="Travel Agent",
        generation_provider=provider,
        model="gemini-2.5-flash",
        instructions="You are a travel agent that helps plan trips."
    )

    # Create an A2A interface for the agent
    a2a = A2AInterface(agent=agent)

Task Management
~~~~~~~~~~~~~

The task management system provides methods for sending, querying, and canceling tasks:

.. code-block:: python

    from agentle.agents.a2a.tasks.task_send_params import TaskSendParams
    from agentle.agents.a2a.tasks.task_query_params import TaskQueryParams
    from agentle.agents.a2a.tasks.task_cancel_params import TaskCancelParams
    from agentle.agents.a2a.message_parts.text_part import TextPart
    from agentle.agents.a2a.messages.message import Message

    # Create a message
    message = Message(
        role="user",
        parts=[TextPart(text="Plan a 3-day trip to Tokyo")]
    )

    # Send a task
    task = a2a.tasks.send(TaskSendParams(message=message))
    task_id = task.id

    # Query a task
    task_result = a2a.tasks.get(TaskQueryParams(id=task_id))

    # Cancel a task
    cancel_result = a2a.tasks.cancel(TaskCancelParams(id=task_id))

Messages and Parts
~~~~~~~~~~~~~~~~

A2A uses a structured message format with different part types:

.. code-block:: python

    from agentle.agents.a2a.messages.message import Message
    from agentle.agents.a2a.message_parts.text_part import TextPart
    from agentle.agents.a2a.message_parts.file_part import FilePart

    # Create a text-only message
    text_message = Message(
        role="user",
        parts=[TextPart(text="Analyze this data")]
    )

    # Create a multimodal message with text and image
    with open("chart.png", "rb") as f:
        image_data = f.read()

    multimodal_message = Message(
        role="user",
        parts=[
            TextPart(text="Analyze this chart:"),
            FilePart(data=image_data, mime_type="image/png")
        ]
    )

Task States
~~~~~~~~~~

The A2A protocol defines several task states for tracking progress:

.. list-table::
   :widths: 30 70
   :header-rows: 1

   * - State
     - Description
   * - ``PENDING``
     - The task has been received but not yet started
   * - ``RUNNING``
     - The task is currently being processed
   * - ``COMPLETED``
     - The task has completed successfully
   * - ``FAILED``
     - The task encountered an error
   * - ``CANCELED``
     - The task was canceled by the client

Task state transitions are handled automatically by the A2A interface.

Exposing A2A via API
------------------

For production use, you can expose your A2A interface as a RESTful API:

.. code-block:: python

    from agentle.agents.a2a.a2a_interface import A2AInterface
    from agentle.agents.agent import Agent
    from agentle.agents.asgi.blacksheep.agent_to_blacksheep_application_adapter import AgentToBlackSheepApplicationAdapter
    from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider

    # Create your agent
    travel_agent = Agent(
        name="Travel Agent",
        description="An AI assistant specialized in planning travel itineraries.",
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.5-flash",
        instructions="""You are a travel agent specialized in creating detailed
        itineraries and providing travel recommendations.""",
    )

    # Create an A2A interface for the agent
    a2a_interface = A2AInterface(agent=travel_agent)

    # Convert the A2A interface to a BlackSheep ASGI application
    app = AgentToBlackSheepApplicationAdapter().adapt(a2a_interface)

    # Run the API server
    if __name__ == "__main__":
        import uvicorn
        uvicorn.run(app, host="127.0.0.1", port=8000)

This creates an API with the following endpoints:

- ``POST /api/v1/tasks/send`` - Send a task to the agent asynchronously
- ``POST /api/v1/tasks/get`` - Get task results
- ``POST /api/v1/tasks/cancel`` - Cancel a running task
- ``WebSocket /api/v1/notifications`` - Subscribe to push notifications about task status changes

Client-Side A2A Communication
---------------------------

To interact with an A2A-compatible service from client code:

.. code-block:: python

    import requests
    import json
    import time

    # Define the A2A service endpoint
    base_url = "http://localhost:8000/api/v1"

    # Create a message
    message = {
        "role": "user",
        "parts": [
            {
                "type": "text",
                "text": "Plan a 3-day trip to Tokyo"
            }
        ]
    }

    # Send a task
    send_response = requests.post(
        f"{base_url}/tasks/send",
        json={"message": message}
    )
    task_id = send_response.json()["id"]
    print(f"Task sent with ID: {task_id}")

    # Poll for results
    while True:
        get_response = requests.post(
            f"{base_url}/tasks/get",
            json={"id": task_id}
        )
        result = get_response.json()
        status = result["result"]["status"]
        
        if status == "COMPLETED":
            print("\nResponse:", result["result"]["history"][1]["parts"][0]["text"])
            break
        elif status == "FAILED":
            print(f"Task failed: {result['result']['error']}")
            break
        print(f"Status: {status}")
        time.sleep(1)

Benefits of A2A
-------------

The A2A protocol offers several advantages:

1. **Interoperability**: Agents built with different frameworks can communicate seamlessly
2. **Enterprise Integration**: Easily integrate agents into existing enterprise applications
3. **Asynchronous Communication**: Non-blocking task management for long-running operations
4. **State Management**: Track task progress and history across agent interactions
5. **Multimodal Support**: Exchange rich content including text, images, and structured data
6. **Open Standard**: Community-driven protocol designed for widespread adoption

Implementation Details
-------------------

Agentle's A2A implementation handles the complexity of:

- **Task Lifecycle Management**: Automatically manages task creation, execution, and state transitions
- **Thread-Safe Execution**: Uses isolated threads with dedicated event loops to prevent concurrency issues
- **Error Handling**: Provides robust error recovery mechanisms during task execution
- **Standardized Messaging**: Offers a clean interface for creating, sending, and processing A2A messages
- **Session Management**: Maintains conversation history and context across multiple interactions
- **Asynchronous Processing**: Transparently converts asynchronous A2A operations into synchronous methods

Advanced A2A Configuration
------------------------

You can customize the A2A interface with additional parameters:

.. code-block:: python

    from agentle.agents.a2a.a2a_interface import A2AInterface, A2AInterfaceOptions

    # Configure the A2A interface with options
    a2a = A2AInterface(
        agent=agent,
        options=A2AInterfaceOptions( # wip
            max_concurrent_tasks=10,  # Maximum number of concurrent tasks
            task_timeout=300,         # Task timeout in seconds
            task_poll_interval=1,     # Polling interval for task status
            keep_completed_tasks=100  # Number of completed tasks to keep in history
        )
    )

Best Practices
------------

1. **Error Handling**: Implement robust error handling for task failures
2. **Timeouts**: Set appropriate timeouts for long-running tasks
3. **Scaling**: Consider using a message broker for high-volume A2A implementations
4. **Authentication**: Add authentication for production A2A endpoints
5. **Monitoring**: Monitor task throughput, completion rates, and error rates