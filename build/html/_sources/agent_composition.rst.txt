=================
Agent Composition
=================

Agentle provides powerful ways to compose multiple agents together to create more complex AI systems. This page explains two main approaches to agent composition: Agent Pipelines and Agent Teams.

Agent Pipelines
-------------

Agent Pipelines connect agents in a sequence where the output of one agent becomes the input to the next. This is useful for breaking down complex tasks into simpler steps that can be handled by specialized agents.

Basic Pipeline Setup
~~~~~~~~~~~~~~~~~~

Here's how to create a basic agent pipeline:

.. code-block:: python

    from agentle.agents.agent import Agent
    from agentle.agents.agent_pipeline import AgentPipeline
    from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider

    # Create a provider for all agents
    provider = GoogleGenerationProvider()

    # Create specialized agents
    research_agent = Agent(
        name="Research Agent",
        generation_provider=provider,
        model="gemini-2.5-flash",
        instructions="""You are a research agent focused on gathering information.
        Be thorough and prioritize accuracy over speculation."""
    )

    analysis_agent = Agent(
        name="Analysis Agent",
        generation_provider=provider,
        model="gemini-2.5-flash",
        instructions="""You are an analysis agent that identifies patterns.
        Highlight meaningful relationships and insights from the data."""
    )

    summary_agent = Agent(
        name="Summary Agent",
        generation_provider=provider,
        model="gemini-2.5-flash",
        instructions="""You are a summary agent that creates concise summaries.
        Present key findings in a logical order with accessible language."""
    )

    # Create a pipeline
    pipeline = AgentPipeline(
        agents=[research_agent, analysis_agent, summary_agent],
    )

    # Run the pipeline
    result = pipeline.run("Research the impact of artificial intelligence on healthcare")
    print(result.text)

How Pipelines Work
~~~~~~~~~~~~~~~~

An Agent Pipeline:

1. Takes the user input and passes it to the first agent
2. Takes the output of the first agent and passes it to the second agent
3. Continues this process through all agents in the pipeline
4. Returns the output of the final agent as the result

By default, each agent in the pipeline receives only the output of the previous agent. If you want to include the original query in each step, you can set the ``include_query`` parameter:

.. code-block:: python

    # Create a pipeline that includes the original query in each step
    pipeline = AgentPipeline(
        agents=[research_agent, analysis_agent, summary_agent],
    )

Pipeline with Different Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~

You can create pipelines with agents that use different models:

.. code-block:: python

    # Agent for broad research (using a more capable but slower model)
    research_agent = Agent(
        name="Research Agent",
        generation_provider=provider,
        model="gemini-2.0-pro",  # More capable model for research
        instructions="You are a thorough research agent that gathers detailed information."
    )

    # Agent for analysis (using a balanced model)
    analysis_agent = Agent(
        name="Analysis Agent",
        generation_provider=provider,
        model="gemini-2.5-flash",  # Balanced model for analysis
        instructions="You identify patterns and insights from the information."
    )

    # Agent for summarization (using a faster model)
    summary_agent = Agent(
        name="Summary Agent",
        generation_provider=provider,
        model="gemini-2.5-flash",  # Fast model for summarization
        instructions="You create concise, clear summaries."
    )

    # Create a pipeline with different models
    pipeline = AgentPipeline(
        agents=[research_agent, analysis_agent, summary_agent]
    )

Agent Teams
---------

Agent Teams consist of multiple specialized agents with an orchestrator that dynamically selects the most appropriate agent for each task. This is useful when you have different agents specialized for different types of tasks.

Basic Team Setup
~~~~~~~~~~~~~~

Here's how to create a basic agent team:

.. code-block:: python

    from agentle.agents.agent import Agent
    from agentle.agents.agent_team import AgentTeam
    from agentle.agents.a2a.models.agent_skill import AgentSkill
    from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider

    # Create a provider for all agents
    provider = GoogleGenerationProvider()

    # Create specialized agents with different skills
    research_agent = Agent(
        name="Research Agent",
        description="Specialized in finding accurate information on various topics",
        generation_provider=provider,
        model="gemini-2.5-flash",
        instructions="You are a research agent focused on gathering accurate information.",
        skills=[
            AgentSkill(name="search", description="Find information on any topic"),
            AgentSkill(name="fact-check", description="Verify factual claims"),
        ],
    )

    coding_agent = Agent(
        name="Coding Assistant",
        description="Specialized in writing and debugging code",
        generation_provider=provider,
        model="gemini-2.5-flash",
        instructions="You are a coding expert focused on writing clean, efficient code.",
        skills=[
            AgentSkill(name="code-generation", description="Write code in various languages"),
            AgentSkill(name="debugging", description="Find and fix bugs in code"),
        ],
    )

    # Create a team with these agents
    team = AgentTeam(
        agents=[research_agent, coding_agent],
        orchestrator_provider=provider,
        orchestrator_model="gemini-2.5-flash",
    )

    # Run the team with different queries
    research_query = "What are the main challenges in quantum computing today?"
    research_result = team.run(research_query)
    print(research_result.text)

    coding_query = "Write a Python function to find the Fibonacci sequence up to n terms."
    coding_result = team.run(coding_query)
    print(coding_result.text)

How Teams Work
~~~~~~~~~~~

An Agent Team:

1. Analyzes the user query through the orchestrator
2. Determines which agent is best suited to handle the query based on skills and descriptions
3. Routes the query to the selected agent
4. Returns the response from the selected agent

The orchestrator can be configured with specific instructions:

.. code-block:: python

    # Create a team with custom orchestrator instructions
    team = AgentTeam(
        agents=[research_agent, coding_agent, math_agent],
        orchestrator_provider=provider,
        orchestrator_model="gemini-2.5-flash",
        orchestrator_instructions="""You are a query router that analyzes user requests
        and determines which specialized agent would be best suited to handle the request.
        Consider the skills and expertise of each agent when making your decision."""
    )

Best Practices
------------

1. **Specialized Instructions**: Make sure each agent in a pipeline or team has clear, specialized instructions
2. **Clear Boundaries**: Ensure clear boundaries between agent responsibilities to avoid overlap
3. **Error Handling**: Consider how errors should propagate through pipelines
4. **Skill Definition**: Define skills clearly to help the orchestrator route queries accurately
