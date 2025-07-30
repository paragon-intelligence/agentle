==========
Deployment
==========

Agentle provides multiple ways to deploy your agents to production environments, from web APIs to interactive UIs. This page covers the available deployment options.

Web API with BlackSheep
----------------------

You can expose your agent or A2A interface as a RESTful API using BlackSheep:

Agent API
~~~~~~~~

Here's how to deploy a simple agent as an API:

.. code-block:: python

    from agentle.agents.agent import Agent
    from agentle.agents.asgi.blacksheep.agent_to_blacksheep_application_adapter import AgentToBlackSheepApplicationAdapter
    from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider

    # Create your agent
    code_assistant = Agent(
        name="Code Assistant",
        description="An AI assistant specialized in helping with programming tasks.",
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.5-flash",
        instructions="""You are a helpful programming assistant.
        You can answer questions about programming languages, help debug code,
        explain programming concepts, and provide code examples.""",
    )

    # Convert the agent to a BlackSheep ASGI application
    app = AgentToBlackSheepApplicationAdapter().adapt(code_assistant)

    # Run the API server
    if __name__ == "__main__":
        import uvicorn
        uvicorn.run(app, host="127.0.0.1", port=8000)

The API will have the following endpoints:

- ``POST /api/v1/agents/code_assistant/run`` - Send prompts to the agent and get responses synchronously
- ``GET /openapi`` - Get the OpenAPI specification
- ``GET /docs`` - Access the interactive API documentation

A2A Interface API
~~~~~~~~~~~~~~

For more complex asynchronous workloads, you can expose your agent using the Agent-to-Agent (A2A) protocol:

.. code-block:: python

    from agentle.agents.a2a.a2a_interface import A2AInterface
    from agentle.agents.agent import Agent
    from agentle.agents.asgi.blacksheep.agent_to_blacksheep_application_adapter import AgentToBlackSheepApplicationAdapter
    from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider

    # Create your agent
    code_assistant = Agent(
        name="Async Code Assistant",
        description="An AI assistant specialized in helping with programming tasks asynchronously.",
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.5-flash",
        instructions="""You are a helpful programming assistant.
        You can answer questions about programming languages, help debug code,
        explain programming concepts, and provide code examples.""",
    )

    # Create an A2A interface for the agent
    a2a_interface = A2AInterface(agent=code_assistant)

    # Convert the A2A interface to a BlackSheep ASGI application
    app = AgentToBlackSheepApplicationAdapter().adapt(a2a_interface)

    # Run the API server
    if __name__ == "__main__":
        import uvicorn
        uvicorn.run(app, host="127.0.0.1", port=8000)

The A2A API will have the following endpoints:

- ``POST /api/v1/tasks/send`` - Send a task to the agent asynchronously
- ``POST /api/v1/tasks/get`` - Get task results
- ``POST /api/v1/tasks/cancel`` - Cancel a running task
- ``WebSocket /api/v1/notifications`` - Subscribe to push notifications about task status changes
- ``GET /openapi`` - Get the OpenAPI specification
- ``GET /docs`` - Access the interactive API documentation

The A2A interface provides a message broker pattern for task processing, similar to RabbitMQ, but exposed through a RESTful API interface.

Interactive UI with Streamlit
---------------------------

Create a chat interface for your agent using Streamlit:

.. code-block:: python

    from agentle.agents.agent import Agent
    from agentle.agents.ui.streamlit import AgentToStreamlit
    from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider

    # Create your agent
    travel_agent = Agent(
        name="Travel Guide",
        description="A helpful travel guide that answers questions about destinations.",
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.5-flash",
        instructions="""You are a knowledgeable travel guide who helps users plan trips.""",
    )

    # Convert the agent to a Streamlit app
    streamlit_app = AgentToStreamlit(
        title="Travel Assistant",
        description="Ask me anything about travel destinations and planning!",
        initial_mode="presentation",  # Can be "dev" or "presentation"
    ).adapt(travel_agent)

    # Run the Streamlit app
    if __name__ == "__main__":
        streamlit_app()

Running the app:

.. code-block:: bash

    streamlit run travel_app.py

The Streamlit interface provides:

1. A clean chat UI for interacting with your agent
2. Message history persistence within the session
3. Ability to clear chat history
4. Dev mode for seeing raw responses and debugging

Custom Integrations
-----------------

For more complex applications, you can directly integrate Agentle agents into your codebase:

Flask Integration
~~~~~~~~~~~~~~

.. code-block:: python

    from flask import Flask, request, jsonify
    from agentle.agents.agent import Agent
    from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider

    app = Flask(__name__)

    # Create your agent
    assistant = Agent(
        name="Flask Assistant",
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.5-flash",
        instructions="You are a helpful assistant integrated with a Flask application."
    )

    @app.route('/api/chat', methods=['POST'])
    def chat():
        user_input = request.json.get('message', '')
        if not user_input:
            return jsonify({'error': 'No message provided'}), 400
            
        # Run the agent
        response = assistant.run(user_input)
        
        # Return the response
        return jsonify({
            'response': response.text,
            # Optionally include other response data
            'raw': response.raw if hasattr(response, 'raw') else None
        })

    if __name__ == '__main__':
        app.run(debug=True)

FastAPI Integration
~~~~~~~~~~~~~~~~

.. code-block:: python

    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
    from agentle.agents.agent import Agent
    from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider

    app = FastAPI()

    # Define request and response models
    class ChatRequest(BaseModel):
        message: str

    class ChatResponse(BaseModel):
        response: str

    # Create your agent
    assistant = Agent(
        name="FastAPI Assistant",
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.5-flash",
        instructions="You are a helpful assistant integrated with a FastAPI application."
    )

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest):
        if not request.message:
            raise HTTPException(status_code=400, detail="No message provided")
            
        # Run the agent
        response = assistant.run(request.message)
        
        # Return the response
        return ChatResponse(response=response.text)

    if __name__ == "__main__":
        import uvicorn
        uvicorn.run(app, host="127.0.0.1", port=8000)

Production Considerations
-----------------------

When deploying Agentle agents to production, consider the following:

Scaling
~~~~~~

- Consider running multiple instances behind a load balancer for high-traffic applications
- For A2A implementations, use a proper message broker (e.g., Redis, RabbitMQ) for task queue management
- Use server-side caching strategies to reduce repeated model calls

Security
~~~~~~~

- Implement proper authentication for API endpoints
- Consider rate limiting to prevent abuse
- Be mindful of the data sent to external LLM providers

Cost Management
~~~~~~~~~~~~

- Monitor and log usage metrics to track costs
- Consider implementing caching strategies for common queries
- Use appropriate model size/type based on complexity requirements

Monitoring
~~~~~~~~

- Implement logging for requests and responses
- Set up error alerting
- Use Agentle's observability features to track performance and usage

Deployment Environment
~~~~~~~~~~~~~~~~~~~

- Use a production-grade ASGI server like Uvicorn or Hypercorn behind a reverse proxy like Nginx
- Deploy using containerization (Docker) for consistency across environments
- Consider serverless deployment options for scalable, on-demand usage