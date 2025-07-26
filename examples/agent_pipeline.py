"""
Agent Pipeline Example

This example demonstrates how to create and use an agent pipeline in the Agentle framework.
A pipeline is a sequence of agents where the output from one agent becomes the input to the next.
"""

from agentle.agents.agent import Agent
from agentle.agents.agent_pipeline import AgentPipeline
from agentle.generations.providers.google.google_generation_provider import (
    GoogleGenerationProvider,
)


# Create a common provider for all agents
provider = GoogleGenerationProvider()

# Step 1: Create specialized agents for each stage of the pipeline

# Research agent that gathers information
research_agent = Agent(
    name="Research Agent",
    generation_provider=provider,
    model="gemini-2.5-flash",
    instructions="""You are a research agent focused on gathering accurate information.
    Your task is to collect relevant facts and data on the given topic.
    Be thorough and prioritize accuracy over speculation.
    ONLY present factual information. Do not analyze, just gather relevant data.
    """,
)

# Analysis agent that processes the research and identifies patterns/insights
analysis_agent = Agent(
    name="Analysis Agent",
    generation_provider=provider,
    model="gemini-2.5-flash",
    instructions="""You are an analysis agent that processes information and identifies patterns.
    You will receive research data and your task is to:
    1. Identify key themes and patterns
    2. Highlight meaningful relationships and correlations
    3. Determine what insights can be drawn from the data
    Focus on analysis, not summarization. Don't repeat all the research, transform it.
    """,
)

# Summary agent that creates a concise, reader-friendly summary
summary_agent = Agent(
    name="Summary Agent",
    generation_provider=provider,
    model="gemini-2.5-flash",
    instructions="""You are a summary agent that creates concise, well-structured summaries.
    Your task is to take the analysis you receive and create a clear, reader-friendly summary that:
    1. Presents key findings in a logical order
    2. Uses simple, accessible language
    3. Maintains accuracy while eliminating unnecessary details
    4. Includes a brief conclusion with the most important takeaways
    """,
)

# Step 2: Create a pipeline of these agents
research_pipeline = AgentPipeline(
    agents=[research_agent, analysis_agent, summary_agent],
    debug_mode=True,  # Enable debug mode to see intermediate steps
)

# Step 3: Run the pipeline with an initial input
TOPIC = "The impact of artificial intelligence on healthcare"
print(f"Running research pipeline on topic: {TOPIC}\n")

result = research_pipeline.run(f"Research the following topic: {TOPIC}")

print("\nFINAL RESULT FROM PIPELINE:")
print("-" * 80)
print(result.text)
print("-" * 80)
