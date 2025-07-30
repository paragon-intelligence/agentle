=============
Observability
=============

Agentle provides built-in observability features to help you monitor, debug, and analyze your agents' performance and behavior in production. This page explains how to set up and use these capabilities.

Basic Tracing Setup
-----------------

Agentle integrates with Langfuse for tracing, which allows you to monitor your agents' performance, behavior, and cost:

.. code-block:: python

    import os
    from agentle.generations.tracing.langfuse import LangfuseObservabilityClient
    from agentle.agents.agent import Agent
    from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider

    # Create a tracing client
    tracing_client = LangfuseObservabilityClient(
        public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
        host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")  # Optional: Default is cloud.langfuse.com
    )

    # Create a generation provider with tracing enabled
    provider = GoogleGenerationProvider(
        api_key=os.environ.get("GOOGLE_API_KEY"),
        tracing_client=tracing_client
    )

    # Create an agent with the traced provider
    agent = Agent(
        name="Traceable Agent",
        generation_provider=provider,
        model="gemini-2.5-flash",
        instructions="You are a helpful assistant."
    )

    # Run the agent - tracing happens automatically
    response = agent.run(
        "What's the weather in Tokyo?", 
        trace_params={
            "name": "weather_query",
            "user_id": "user123",
            "metadata": {"source": "mobile_app"}
        }
    )

What Gets Traced
--------------

When you enable tracing, Agentle automatically captures:

1. **Model calls**: Details about each generation request, including:
   - Model name and parameters
   - Input tokens and cost
   - Output tokens and cost
   - Latency
   
2. **Tool executions**: When tools are called, including:
   - Tool name
   - Input parameters
   - Output results
   - Execution time
   
3. **User interactions**: Complete conversation history for context

4. **Metadata**: Custom metadata you provide via trace_params


Tracing Pipelines and Teams
-------------------------

When using agent compositions, Agentle automatically traces the entire workflow:

.. code-block:: python

    from agentle.agents.agent_pipeline import AgentPipeline
    from agentle.agents.agent_team import AgentTeam

    # Create a pipeline with tracing
    pipeline = AgentPipeline(
        agents=[research_agent, analysis_agent, summary_agent],
        debug_mode=True
    )

    # Run the pipeline with trace parameters
    result = pipeline.run(
        "Research quantum computing advances", 
        trace_params={
            "name": "research_pipeline",
            "user_id": "researcher_001",
            "metadata": {"department": "physics", "priority": "high"}
        }
    )

    # Create a team with tracing
    team = AgentTeam(
        agents=[research_agent, coding_agent, math_agent],
        orchestrator_provider=provider,  # Using the provider with tracing enabled
        orchestrator_model="gemini-2.5-flash"
    )

    # Run the team with trace parameters
    result = team.run(
        "Explain the mathematical foundations of quantum computing",
        trace_params={
            "name": "quantum_research",
            "user_id": "researcher_001"
        }
    )

Monitoring in Production
----------------------

For production deployments, you can set up comprehensive monitoring:

.. code-block:: python

    # API endpoint with tracing
    @app.post("/api/chat")
    async def chat(request: ChatRequest):
        # Extract user info from request
        user_id = request.user_id if hasattr(request, "user_id") else "anonymous"
        
        # Run the agent with tracing
        response = agent.run(
            request.message,
            trace_params={
                "name": "api_chat_request",
                "user_id": user_id,
                "metadata": {
                    "endpoint": "/api/chat",
                    "client_ip": request.client.host,
                    "request_id": str(uuid.uuid4())
                }
            }
        )
        
        # Return the response
        return ChatResponse(response=response.text)

Setting Up Langfuse
-----------------

To use Langfuse for tracing:

1. **Sign up**: Create an account at `Langfuse <https://cloud.langfuse.com>`_
2. **Get credentials**: Obtain your Public Key and Secret Key from the Langfuse dashboard
3. **Set environment variables**:

   .. code-block:: bash

       export LANGFUSE_PUBLIC_KEY="your-public-key"
       export LANGFUSE_SECRET_KEY="your-secret-key"
       export LANGFUSE_HOST="https://cloud.langfuse.com"  # Optional

4. **Install dependencies**:

   .. code-block:: bash

       pip install langfuse

5. **Initialize the client** in your application as shown above

The Langfuse dashboard provides:

- Real-time monitoring of all agent interactions
- Cost tracking and usage analytics
- Performance metrics and bottleneck identification
- Conversation history and context inspection
- Error rate monitoring
- Custom filtering and searching

Here's an example of what production traces look like in Langfuse:

.. image:: /../../docs/langfuse_traces.png
   :alt: Langfuse Traces Example
   :width: 800
   :align: center

Customizing Observability
-----------------------

You can implement custom observability clients by implementing the ``StatefulObservabilityClient`` abstract base class. Take a look at internal code to see how it works.


Best Practices
------------

1. **Consistent Naming**: Use consistent naming conventions for traces and spans
2. **Meaningful Metadata**: Include relevant metadata for filtering and analysis
3. **User Identification**: Always include user IDs when possible for user-centric analysis
4. **Appropriate Detail Level**: Balance between too much and too little information
5. **Error Tracking**: Ensure errors are properly captured and categorized
6. **Regular Analysis**: Review traces regularly to identify patterns and issues