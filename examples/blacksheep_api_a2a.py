"""
Blacksheep A2A API Example

This example demonstrates how to expose an agent with A2A (Agent-to-Agent) interface
as a web API using the Blacksheep ASGI framework. This creates a RESTful API that
allows applications to interact with your agent asynchronously over HTTP.
"""

from agentle.agents.a2a.a2a_interface import A2AInterface
from agentle.agents.agent import Agent
from agentle.agents.asgi.blacksheep.agent_to_blacksheep_application_adapter import (
    AgentToBlackSheepApplicationAdapter,
)
from agentle.generations.providers.google.google_genai_generation_provider import (
    GoogleGenaiGenerationProvider,
)

# Create a simple agent
code_assistant = Agent(
    name="Async Code Assistant",
    description="An AI assistant specialized in helping with programming tasks asynchronously.",
    generation_provider=GoogleGenaiGenerationProvider(),
    model="gemini-2.0-flash",
    instructions="""You are a helpful programming assistant.
    You can answer questions about programming languages, help debug code,
    explain programming concepts, and provide code examples.
    Your responses should be clear, concise, and technically accurate.""",
)

# Create an A2A interface for the agent
a2a_interface = A2AInterface(agent=code_assistant)

# Convert the A2A interface to a BlackSheep ASGI application
app = AgentToBlackSheepApplicationAdapter().adapt(a2a_interface)

# This is how you would run the API server
if __name__ == "__main__":
    # Run the server (will be available at http://localhost:8000)
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)

"""
To run this example:
1. Install required packages: pip install uvicorn blacksheep openapidocs
2. Run: python blacksheep_api_a2a.py
3. The API will be available at http://localhost:8000
4. Access the API documentation at http://localhost:8000/docs

API Endpoints:
- POST /api/v1/tasks/send: Send a task to the agent asynchronously
- POST /api/v1/tasks/get: Get task results
- POST /api/v1/tasks/cancel: Cancel a running task
- WebSocket /api/v1/notifications: Subscribe to push notifications

Example API calls:

1. Send a task:
```
curl -X POST http://localhost:8000/api/v1/tasks/send \\
  -H "Content-Type: application/json" \\
  -d '{
    "task_params": {
      "message": {
        "role": "user",
        "parts": [{"type": "text", "text": "How do I create a simple web server in Python?"}]
      },
      "sessionId": "session-1"
    }
  }'
```

2. Get task results:
```
curl -X POST http://localhost:8000/api/v1/tasks/get \\
  -H "Content-Type: application/json" \\
  -d '{
    "query_params": {
      "id": "task-id-from-send-response"
    }
  }'
```

3. Cancel a task:
```
curl -X POST http://localhost:8000/api/v1/tasks/cancel \\
  -H "Content-Type: application/json" \\
  -d '{
    "task_id": "task-id-from-send-response"
  }'
```

4. For WebSocket notifications, use a WebSocket client to connect to:
   ws://localhost:8000/api/v1/notifications
"""
