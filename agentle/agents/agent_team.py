from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast, Optional
from uuid import UUID

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.agent import Agent
from agentle.agents.agent_config import AgentConfig
from agentle.agents.agent_input import AgentInput
from agentle.agents.agent_run_output import AgentRunOutput
from agentle.generations.providers.base.generation_provider import GenerationProvider
from agentle.generations.providers.google.google_genai_generation_provider import (
    GoogleGenaiGenerationProvider,
)
from rsb.coroutines.run_sync import run_sync


class OrchestratorOutput(BaseModel):
    """Structured output for the orchestrator agent"""

    agent_id: UUID  # UUID of the chosen agent
    task_done: bool = False  # Indicates if the task is complete


class AgentTeam(BaseModel):
    agents: Sequence[Agent[Any]]
    orchestrator_provider: GenerationProvider = Field(
        default_factory=GoogleGenaiGenerationProvider
    )
    orchestrator_model: str
    team_config: AgentConfig = Field(default_factory=AgentConfig)

    def run(self, input: AgentInput) -> AgentRunOutput[Any]:
        return run_sync(self.run_async, input=input)

    async def run_async(self, input: AgentInput) -> AgentRunOutput[Any]:
        """Analyse the task. Call the appropriate agents until the task is concluded.
        Remove the called agents from the call stack. Simple implementation for now."""
        if not self.agents:
            raise ValueError("AgentTeam must have at least one agent")

        # Build a detailed description of available agents for the orchestrator
        agent_descriptions: list[str] = []
        for agent in self.agents:
            # Create a description of each agent's capabilities
            skills_desc = (
                ", ".join([skill.name for skill in agent.skills])
                if agent.skills
                else "No specific skills defined"
            )

            agent_desc = f"""
            Agent ID: {agent.uid}
            Name: {agent.name}
            Description: {agent.description}
            Skills: {skills_desc}
            """
            agent_descriptions.append(agent_desc)

        agents_info = "\n".join(agent_descriptions)

        # Create an orchestrator agent with knowledge of available agents
        orchestrator_agent = Agent(
            name="Orchestrator",
            generation_provider=self.orchestrator_provider,
            model=self.orchestrator_model,
            instructions=f"""You are an orchestrator agent that analyzes tasks and determines which agent
            should handle them. Examine the input carefully and select the most appropriate agent based on 
            its capabilities and expertise. You should also determine if the task is complete.
            
            Here are the available agents you can choose from:
            {agents_info}
            
            For each task you are given, you must:
            1. Analyze the task requirements thoroughly
            2. Select the most appropriate agent by its ID based on its capabilities
            3. Determine if the task is complete (set task_done to true if it is)
            
            If you believe the task has been fully addressed, set task_done to true.
            If you believe the task requires further processing, select the appropriate agent and set task_done to false.
            """,
            response_schema=OrchestratorOutput,
            config=self.team_config,  # Use the team config for the orchestrator
        )

        # Create agent lookup by UUID
        agent_map = {str(agent.uid): agent for agent in self.agents}

        # Initial context with the original input
        current_input = input
        task_done = False

        # Keep track of the last output to return
        last_output: Optional[AgentRunOutput[Any]] = None

        # Use maxIterations from team_config to prevent infinite loops
        iteration_count = 0
        max_iterations = self.team_config.maxIterations

        # Keep track of the conversation history for context
        conversation_history: list[str] = []

        # Iterate until the task is done or max iterations is reached
        while not task_done and iteration_count < max_iterations:
            # Increment iteration counter
            iteration_count += 1

            # Format orchestrator input with task history
            orchestrator_input = current_input
            if conversation_history:
                # If we have history, add it as context
                history_text = "\n\n".join(conversation_history)
                if isinstance(current_input, str):
                    orchestrator_input = f"""
Task Context/History:
{history_text}

Current input:
{current_input}
"""

            # Use the orchestrator to decide which agent should handle the task
            orchestrator_result = await orchestrator_agent.run_async(orchestrator_input)
            orchestrator_output = orchestrator_result.parsed

            if not orchestrator_output:
                # Fallback in case the orchestrator fails to produce structured output
                return await self.agents[0].run_async(current_input)

            # Check if the task is done
            task_done = orchestrator_output.task_done
            if task_done:
                return (
                    last_output
                    if last_output is not None
                    else await self.agents[0].run_async(current_input)
                )

            # Get the chosen agent
            agent_id = str(orchestrator_output.agent_id)
            if agent_id not in agent_map:
                # Fallback if the orchestrator chooses an unknown agent
                return await self.agents[0].run_async(current_input)

            chosen_agent = agent_map[agent_id]

            # Run the chosen agent with the current input
            agent_output = await chosen_agent.run_async(current_input)
            last_output = agent_output

            # Update the conversation history
            if isinstance(current_input, str):
                conversation_history.append(f"User/Task: {current_input}")
            if agent_output.generation.text:
                conversation_history.append(
                    f"Agent '{chosen_agent.name}': {agent_output.generation.text}"
                )

            # Update the input with the agent's response for the next iteration
            if agent_output.generation.text:
                current_input = agent_output.generation.text
            else:
                # If we don't have text output, use the original input again
                return agent_output

        # If we've reached max iterations, return the last output
        if iteration_count >= max_iterations:
            print(
                f"Warning: AgentTeam reached maximum iterations ({max_iterations}) without completion"
            )

        # This should never be reached due to the returns above, but added for type safety
        if last_output is None:
            return await self.agents[0].run_async(input)
        return last_output

    def __add__(self, other: Agent[Any] | AgentTeam) -> AgentTeam:
        if isinstance(other, Agent):
            return AgentTeam(
                agents=cast(Sequence[Agent[Any]], list(self.agents) + [other]),
                orchestrator_provider=self.orchestrator_provider,
                orchestrator_model=self.orchestrator_model,
                team_config=self.team_config,
            )

        return AgentTeam(
            agents=cast(Sequence[Agent[Any]], list(self.agents) + list(other.agents)),
            orchestrator_provider=self.orchestrator_provider,
            orchestrator_model=self.orchestrator_model,
            team_config=self.team_config,
        )
