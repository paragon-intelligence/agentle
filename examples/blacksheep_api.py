"""
Blacksheep API Example

This example demonstrates how to expose an agent as a web API using the Blacksheep ASGI framework.
This creates a RESTful API that allows applications to interact with your agent over HTTP.
"""

from agentle.agents.agent import Agent
from agentle.agents.asgi.blacksheep.agent_to_blacksheep_application_adapter import (
    AgentToBlackSheepApplicationAdapter,
)
from agentle.generations.providers.google.google_genai_generation_provider import (
    GoogleGenaiGenerationProvider,
)

generation_provider = GoogleGenaiGenerationProvider()

# Create a simple agent
code_assistant = Agent(
    name="Code Assistant",
    description="An AI assistant specialized in helping with programming tasks.",
    generation_provider=generation_provider,
    model="gemini-2.0-flash",
    instructions="""You are a helpful programming assistant.
    You can answer questions about programming languages, help debug code,
    explain programming concepts, and provide code examples.
    Your responses should be clear, concise, and technically accurate.""",
)

# Convert the agent to a Blacksheep ASGI application
app = AgentToBlackSheepApplicationAdapter().adapt(code_assistant)

# This is how you would run the API server
if __name__ == "__main__":
    # Run the server (will be available at http://localhost:8000)
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)

"""
To run this example:
1. Install required packages: pip install uvicorn blacksheep openapidocs
2. Run: python blacksheep_api.py
3. The API will be available at http://localhost:8000
4. Access the API documentation at http://localhost:8000/docs

API Endpoints:
- POST /agent/generate: Send prompts to the agent and get responses
- GET /openapi: Get the OpenAPI specification
- GET /docs: Access the interactive API documentation

Example API call:
```
curl -X POST http://localhost:8000/api/v1/agents/agent/run \\
  -H "Content-Type: application/json" \\
  -d '{"prompt": "How do I create a simple web server in Python?"}'
```
"""
