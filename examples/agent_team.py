"""
Agent Team Example

This example demonstrates how to create and use an agent team in the Agentle framework.
A team consists of specialized agents coordinated by an orchestrator that dynamically
selects the most appropriate agent for each task.
"""

from agentle.agents.agent import Agent
from agentle.agents.agent_team import AgentTeam
from agentle.agents.agent_config import AgentConfig
from agentle.generations.providers.google.google_genai_generation_provider import (
    GoogleGenaiGenerationProvider,
)
from agentle.agents.a2a.models.agent_skill import AgentSkill

# Create a common provider for all agents
provider = GoogleGenaiGenerationProvider()

# Step 1: Create specialized agents with different skills

# Research agent specialized in finding information
research_agent = Agent(
    name="Research Agent",
    description="Specialized in finding accurate information and data on various topics",
    generation_provider=provider,
    model="gemini-2.0-flash",
    instructions="""You are a research agent focused on gathering accurate information.
    Your task is to find and present factual information from reliable sources.
    Always prioritize accuracy over speculation. Cite sources when possible.
    """,
    skills=[
        AgentSkill(name="search", description="Find information on any topic"),
        AgentSkill(name="fact-check", description="Verify factual claims"),
    ],
)

# Coding agent specialized in writing and debugging code
coding_agent = Agent(
    name="Coding Agent",
    description="Specialized in writing and debugging code in multiple programming languages",
    generation_provider=provider,
    model="gemini-2.0-flash",
    instructions="""You are a coding expert focused on writing clean, efficient code.
    You can create code in various languages, debug existing code, and explain
    code functionality. Always provide well-commented solutions.
    """,
    skills=[
        AgentSkill(
            name="code-generation", description="Write code in various languages"
        ),
        AgentSkill(name="debugging", description="Find and fix bugs in code"),
    ],
)

# Writing agent specialized in creating content
writing_agent = Agent(
    name="Writing Agent",
    description="Specialized in creating clear, engaging written content in various styles",
    generation_provider=provider,
    model="gemini-2.0-flash",
    instructions="""You are a writing expert focused on creating high-quality content.
    You can write in various styles and formats, from technical documentation
    to creative content. Always focus on clarity and engagement.
    """,
    skills=[
        AgentSkill(name="content-creation", description="Create written content"),
        AgentSkill(name="editing", description="Improve existing written content"),
    ],
)

# Step 2: Create a team with these agents
team = AgentTeam(
    agents=[research_agent, coding_agent, writing_agent],
    orchestrator_provider=provider,
    orchestrator_model="gemini-2.0-flash",
    team_config=AgentConfig(maxIterations=10),
)

# Step 3: Run the team with queries that require different skills

print("Running agent team with various queries...\n")

# A query requiring research skills
research_query = "What are the main challenges in quantum computing today?"
print(f"QUERY: {research_query}")
research_result = team.run(research_query)
print(f"RESPONSE: {research_result.text[:500]}...\n")

# A query requiring coding skills
coding_query = "Write a Python function to find the Fibonacci sequence up to n terms."
print(f"QUERY: {coding_query}")
coding_result = team.run(coding_query)
print(f"RESPONSE: {coding_result.text[:500]}...\n")

# A query requiring both research and writing skills
combined_query = (
    "Explain machine learning concepts for beginners in a clear, engaging way."
)
print(f"QUERY: {combined_query}")
combined_result = team.run(combined_query)
print(f"RESPONSE: {combined_result.text[:500]}...\n")
