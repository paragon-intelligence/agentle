"""
Agent Team Module for the Agentle Framework

This module provides functionality for creating dynamic teams of AI agents managed by an orchestrator.
AgentTeam implements a flexible approach to multi-agent systems where agents are selected at runtime
based on the specific requirements of each task and subtask.

Unlike the AgentPipeline which follows a fixed sequence of agents, AgentTeam uses an orchestrator
agent to dynamically analyze tasks and select the most appropriate agent to handle each step.
This provides greater flexibility and adaptability, particularly for complex workflows where:

1. The optimal sequence of agents depends on the specific task
2. Different agents may be needed depending on the content of previous responses
3. The same agent may need to be invoked multiple times for different aspects of a task
4. The task completion criteria can only be determined at runtime

Example:
```python
from agentle.agents.agent import Agent
from agentle.agents.agent_team import AgentTeam
from agentle.agents.agent_config import AgentConfig
from agentle.generations.providers.google.google_genai_generation_provider import GoogleGenaiGenerationProvider

# Create provider for all agents
provider = GoogleGenaiGenerationProvider()

# Create specialized agents
research_agent = Agent(
    name="Research Agent",
    description="Specialized in finding information and data on various topics",
    generation_provider=provider,
    model="gemini-2.0-flash",
    instructions="You are a research agent focused on gathering accurate information.",
)

coding_agent = Agent(
    name="Coding Agent",
    description="Specialized in writing and debugging code in various languages",
    generation_provider=provider,
    model="gemini-2.0-flash",
    instructions="You are a coding expert that writes clean, efficient code.",
)

writing_agent = Agent(
    name="Writing Agent",
    description="Specialized in creating clear and engaging written content",
    generation_provider=provider,
    model="gemini-2.0-flash",
    instructions="You are a writing expert that creates compelling content.",
)

# Create a team with these agents and an orchestrator
team = AgentTeam(
    agents=[research_agent, coding_agent, writing_agent],
    orchestrator_provider=provider,
    orchestrator_model="gemini-2.0-flash",
    team_config=AgentConfig(maxIterations=10)
)

# Run the team with a task
result = team.run("Research the latest advancements in quantum computing and summarize them.")
print(result.generation.text)
```
"""

from __future__ import annotations

import logging
from collections.abc import MutableSequence, Sequence
from typing import Any, Optional, Self, cast
from uuid import UUID

from rsb.coroutines.run_sync import run_sync
from rsb.models.base_model import BaseModel
from rsb.models.field import Field
from rsb.models.model_validator import model_validator

from agentle.agents.agent import Agent
from agentle.agents.agent_config import AgentConfig
from agentle.agents.agent_input import AgentInput
from agentle.agents.agent_run_output import AgentRunOutput
from agentle.generations.providers.base.generation_provider import GenerationProvider
from agentle.generations.providers.google.google_genai_generation_provider import (
    GoogleGenaiGenerationProvider,
)

logger = logging.getLogger(__name__)


class _OrchestratorOutput(BaseModel):
    """
    Structured output for the orchestrator agent to determine task routing.

    This model defines the format of responses expected from the orchestrator agent.
    The orchestrator analyzes tasks and determines: (1) which agent should handle
    the current task step, and (2) whether the overall task is complete.

    Attributes:
        agent_id: The str of the chosen agent that should handle the current task step.
        task_done: A boolean flag indicating whether the overall task is complete.
                  When True, the team execution will terminate and return the final result.
    """

    agent_id: str
    task_done: bool

    @model_validator(mode="after")
    def validate_agent_id(self) -> Self:
        if not self.agent_id:
            raise ValueError("agent_id must be a non-empty string")

        # check if it's valid UUID
        try:
            UUID(self.agent_id)
        except ValueError:
            raise ValueError("agent_id must be a valid UUID")

        return self


class AgentTeam(BaseModel):
    """
    A dynamic team of AI agents managed by an orchestrator agent.

    AgentTeam implements an approach to multi-agent systems where agents are selected
    dynamically at runtime by an orchestrator. The orchestrator analyzes each task or
    subtask and determines which specialized agent is best suited to handle it.

    This provides greater flexibility than sequential pipelines, as the exact sequence
    of agent invocations can be determined based on the specifics of the task and the
    content of intermediate responses.

    Key features:
    - Dynamic agent selection: The orchestrator chooses the most appropriate agent for each step
    - Adaptable workflows: The sequence of agents is determined at runtime based on the task
    - Conversation history: Context is maintained throughout the task execution
    - Configurable execution limits: Prevents infinite loops with clear termination criteria

    Attributes:
        agents: A sequence of specialized Agent instances available to the team.
        orchestrator_provider: The generation provider used by the orchestrator agent.
        orchestrator_model: The model to be used by the orchestrator agent.
        team_config: Configuration options for the team, including iteration limits.

    Examples:
        ```python
        # Create a basic team with several specialized agents
        provider = GoogleGenaiGenerationProvider()

        math_agent = Agent(
            name="Math Agent",
            description="Expert in solving mathematical problems",
            generation_provider=provider,
            model="gemini-2.0-flash",
            instructions="You are a mathematics expert."
        )

        language_agent = Agent(
            name="Language Agent",
            description="Expert in language translation and linguistics",
            generation_provider=provider,
            model="gemini-2.0-flash",
            instructions="You are a language and translation expert."
        )

        team = AgentTeam(
            agents=[math_agent, language_agent],
            orchestrator_provider=provider,
            orchestrator_model="gemini-2.0-flash"
        )

        # Run the team on a task
        result = team.run("Translate the phrase 'The square root of 144 is 12' into French")
        ```

        Creating a team with custom configuration:
        ```python
        # Create a team with custom configuration
        custom_team = AgentTeam(
            agents=[agent1, agent2, agent3],
            orchestrator_provider=provider,
            orchestrator_model="gemini-2.0-flash",
            team_config=AgentConfig(
                maxIterations=15,  # Allow more iterations than default
                maxToolCalls=20    # Allow more tool calls if agents use tools
            )
        )

        # Run the team with a complex multi-step task
        result = custom_team.run("Research, analyze, and summarize recent developments in AI safety")
        ```
    """

    agents: Sequence[Agent[Any]] = Field(
        ...,  # This is a required field
        description="A sequence of specialized Agent instances available to the team.",
    )

    orchestrator_provider: GenerationProvider = Field(
        default_factory=GoogleGenaiGenerationProvider,
        description="The generation provider used by the orchestrator agent.",
    )

    orchestrator_model: str = Field(
        ...,  # This is a required field
        description="The model to be used by the orchestrator agent.",
    )

    team_config: AgentConfig = Field(
        default_factory=AgentConfig,
        description="Configuration options for the team, including iteration limits.",
    )

    class Config:
        arbitrary_types_allowed = True

    def extend(self, other: Agent[Any] | AgentTeam) -> AgentTeam:
        """
        Extend the current team with another Agent or AgentTeam.
        """
        return self + other

    async def resume_async(
        self, resumption_token: str, approval_data: dict[str, Any] | None = None
    ) -> AgentRunOutput[Any]:
        """
        Resume a suspended team execution.

        This method resumes a team that was suspended due to a tool requiring
        human approval. It retrieves the team state and continues execution
        from where it left off.

        Args:
            resumption_token: Token from a suspended team execution
            approval_data: Optional approval data to pass to the resumed execution

        Returns:
            AgentRunOutput with the completed or newly suspended execution

        Raises:
            ValueError: If the resumption token is invalid or not for a team
        """
        if not self.agents:
            raise ValueError("AgentTeam must have at least one agent")

        # Resume the agent execution first
        # We need to get the suspended agent and resume it
        # For now, we'll delegate to the first agent's resume method
        resumed_result = await self.agents[0].resume_async(
            resumption_token, approval_data
        )

        # Check if we have team state to continue from
        team_state = resumed_result.context.get_checkpoint_data("team_state")
        if not team_state:
            # If no team state, just return the resumed result
            return resumed_result

        # Extract team state
        iteration_count = team_state.get("iteration_count", 0)
        max_iterations = team_state.get(
            "max_iterations", self.team_config.maxIterations
        )
        current_input = team_state.get("current_input", "")
        conversation_history = team_state.get("conversation_history", [])
        original_input = team_state.get("original_input", "")

        # Recreate agent map
        agent_map = {str(agent.uid): agent for agent in self.agents}

        # Create orchestrator agent (same as in run_async)
        agent_descriptions: MutableSequence[str] = []
        for agent in self.agents:
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
            response_schema=_OrchestratorOutput,
            config=self.team_config,
        )

        # Update conversation history with the resumed result
        if resumed_result.generation and resumed_result.generation.text:
            conversation_history.append(
                f"Resumed execution: {resumed_result.generation.text}"
            )
            current_input = resumed_result.generation.text

        task_done = False
        last_output = resumed_result

        # Continue the team execution loop
        while not task_done and iteration_count < max_iterations:
            iteration_count += 1

            # Format orchestrator input with task history
            orchestrator_input = current_input
            if conversation_history:
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
                logger.warning(
                    "Orchestrator failed to produce structured output, using first agent as fallback"
                )
                return await self.agents[0].run_async(current_input)

            # Check if the task is done
            task_done = orchestrator_output.task_done
            if task_done:
                logger.info(
                    f"Task marked as complete by orchestrator after {iteration_count} iterations (resumed)"
                )
                return last_output

            # Get the chosen agent
            agent_id = str(orchestrator_output.agent_id)
            if agent_id not in agent_map:
                logger.warning(
                    f"Orchestrator selected unknown agent ID: {agent_id}, using first agent as fallback"
                )
                return await self.agents[0].run_async(current_input)

            chosen_agent = agent_map[agent_id]
            logger.info(
                f"Resumed iteration {iteration_count}: Orchestrator selected agent '{chosen_agent.name}'"
            )

            # Run the chosen agent with the current input
            agent_output = await chosen_agent.run_async(current_input)
            last_output = agent_output

            # Check if the agent execution was suspended again
            if agent_output.is_suspended:
                suspension_msg = (
                    f"Team execution suspended again at iteration {iteration_count} "
                    + f"(Agent '{chosen_agent.name}'): {agent_output.suspension_reason}"
                )
                logger.info(suspension_msg)

                # Update team state for the new suspension point
                if agent_output.context:
                    agent_output.context.set_checkpoint_data(
                        "team_state",
                        {
                            "iteration_count": iteration_count,
                            "max_iterations": max_iterations,
                            "current_input": current_input,
                            "conversation_history": conversation_history,
                            "agent_map": {
                                str(uid): agent.name for uid, agent in agent_map.items()
                            },
                            "last_selected_agent_id": agent_id,
                            "original_input": original_input,
                        },
                    )

                return agent_output

            # Update the conversation history
            if isinstance(current_input, str):
                conversation_history.append(f"User/Task: {current_input}")
            if agent_output.generation and agent_output.generation.text:
                conversation_history.append(
                    f"Agent '{chosen_agent.name}': {agent_output.generation.text}"
                )

            # Update the input with the agent's response for the next iteration
            if agent_output.generation and agent_output.generation.text:
                current_input = agent_output.generation.text
            else:
                logger.warning(
                    f"Agent '{chosen_agent.name}' produced no text output, returning current output"
                )
                return agent_output

        # If we've reached max iterations, return the last output
        if iteration_count >= max_iterations:
            logger.warning(
                f"Warning: AgentTeam reached maximum iterations ({max_iterations}) without completion (resumed)"
            )

        return last_output

    def resume(
        self, resumption_token: str, approval_data: dict[str, Any] | None = None
    ) -> AgentRunOutput[Any]:
        """
        Resume a suspended team execution synchronously.

        Args:
            resumption_token: Token from a suspended team execution
            approval_data: Optional approval data to pass to the resumed execution

        Returns:
            AgentRunOutput with the completed or newly suspended execution
        """
        return run_sync(
            self.resume_async,
            resumption_token=resumption_token,
            approval_data=approval_data,
        )

    def run(self, input: AgentInput | Any) -> AgentRunOutput[Any]:
        """
        Run the agent team synchronously with the provided input.

        This method is a synchronous wrapper around run_async that allows
        the team to be used in synchronous contexts.

        Args:
            input: The initial input to the team.
                  Can be a string, UserMessage, Context, or any supported AgentInput type.

        Returns:
            AgentRunOutput: The final output from the team after task completion.

        Example:
            ```python
            team = AgentTeam(
                agents=[research_agent, writing_agent, code_agent],
                orchestrator_provider=provider,
                orchestrator_model="gemini-2.0-flash"
            )

            # Simple string input
            result = team.run("Create a Python function to calculate the Fibonacci sequence")

            # Access the result
            final_text = result.generation.text
            ```
        """
        return run_sync(self.run_async, input=input)

    async def run_async(self, input: AgentInput | Any) -> AgentRunOutput[Any]:
        """
        Dynamically executes a team of agents guided by an orchestrator.

        This method creates an orchestrator agent that analyzes each task step and
        selects the most appropriate agent to handle it. The process continues iteratively
        until the orchestrator determines the task is complete or the maximum number of
        iterations is reached.

        Process flow:
        1. Orchestrator analyzes the current input/task
        2. Orchestrator selects the most appropriate agent and indicates if task is complete
        3. If task is complete, return the latest result
        4. Otherwise, the selected agent processes the input
        5. The agent's output becomes the next input
        6. The process repeats until completion or max iterations

        Args:
            input: The initial input to the team

        Returns:
            AgentRunOutput: The final result when the task is complete

        Raises:
            ValueError: If the team contains no agents

        Example:
            ```python
            # Using the team with async/await
            import asyncio

            async def process_complex_task():
                team = AgentTeam(
                    agents=[agent1, agent2, agent3],
                    orchestrator_provider=provider,
                    orchestrator_model="gemini-2.0-flash"
                )

                result = await team.run_async("Perform this complex multi-step task")
                return result.generation.text

            final_result = asyncio.run(process_complex_task())
            ```

        Notes:
            - The team_config.maxIterations parameter controls the maximum number of
              agent invocations to prevent infinite loops.
            - Conversation history is maintained to provide context to the orchestrator.
            - The orchestrator has complete information about all available agents.
        """
        if not self.agents:
            raise ValueError("AgentTeam must have at least one agent")

        # Build a detailed description of available agents for the orchestrator
        agent_descriptions: MutableSequence[str] = []
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
            response_schema=_OrchestratorOutput,
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
        conversation_history: MutableSequence[str] = []

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
                logger.warning(
                    "Orchestrator failed to produce structured output, using first agent as fallback"
                )
                return await self.agents[0].run_async(current_input)

            # Check if the task is done
            task_done = orchestrator_output.task_done
            if task_done:
                logger.info(
                    f"Task marked as complete by orchestrator after {iteration_count} iterations"
                )
                return (
                    last_output
                    if last_output is not None
                    else await self.agents[0].run_async(current_input)
                )

            # Get the chosen agent
            agent_id = str(orchestrator_output.agent_id)
            if agent_id not in agent_map:
                # Fallback if the orchestrator chooses an unknown agent
                logger.warning(
                    f"Orchestrator selected unknown agent ID: {agent_id}, using first agent as fallback"
                )
                return await self.agents[0].run_async(current_input)

            chosen_agent = agent_map[agent_id]
            logger.info(
                f"Iteration {iteration_count}: Orchestrator selected agent '{chosen_agent.name}'"
            )

            # Run the chosen agent with the current input
            agent_output = await chosen_agent.run_async(current_input)
            last_output = agent_output

            # Check if the agent execution was suspended
            if agent_output.is_suspended:
                suspension_msg = (
                    f"Team execution suspended at iteration {iteration_count} "
                    + f"(Agent '{chosen_agent.name}'): {agent_output.suspension_reason}"
                )
                logger.info(suspension_msg)

                # Store team state in the context for resumption
                if agent_output.context:
                    agent_output.context.set_checkpoint_data(
                        "team_state",
                        {
                            "iteration_count": iteration_count,
                            "max_iterations": max_iterations,
                            "current_input": current_input,
                            "conversation_history": conversation_history,
                            "agent_map": {
                                str(uid): agent.name for uid, agent in agent_map.items()
                            },
                            "last_selected_agent_id": agent_id,
                            "original_input": input,
                        },
                    )

                # Return the suspended output immediately
                return agent_output

            # Update the conversation history
            if isinstance(current_input, str):
                conversation_history.append(f"User/Task: {current_input}")
            if agent_output.generation and agent_output.generation.text:
                conversation_history.append(
                    f"Agent '{chosen_agent.name}': {agent_output.generation.text}"
                )

            # Update the input with the agent's response for the next iteration
            if agent_output.generation and agent_output.generation.text:
                current_input = agent_output.generation.text
            else:
                # If we don't have text output, use the original input again
                logger.warning(
                    f"Agent '{chosen_agent.name}' produced no text output, returning current output"
                )
                return agent_output

        # If we've reached max iterations, return the last output
        if iteration_count >= max_iterations:
            logger.warning(
                f"Warning: AgentTeam reached maximum iterations ({max_iterations}) without completion"
            )

        # This should never be reached due to the returns above, but added for type safety
        if last_output is None:
            return await self.agents[0].run_async(input)
        return last_output

    def __add__(self, other: Agent[Any] | AgentTeam) -> AgentTeam:
        """
        Combine this AgentTeam with another Agent or AgentTeam.

        This operator allows for easy composition of teams by combining their agents.
        When adding an Agent, it's incorporated into the current team's agent list.
        When adding another AgentTeam, all its agents are added to the current team.

        Args:
            other: Another Agent or AgentTeam to combine with this team

        Returns:
            AgentTeam: A new AgentTeam containing all agents from both sources

        Example:
            ```python
            # Create initial team
            basic_team = AgentTeam(
                agents=[agent1, agent2],
                orchestrator_provider=provider,
                orchestrator_model="gemini-2.0-flash"
            )

            # Add another agent
            expanded_team = basic_team + specialized_agent

            # Add another team
            other_team = AgentTeam(agents=[agent3, agent4], ...)
            combined_team = expanded_team + other_team

            # The combined team now contains all agents from both teams
            ```
        """
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
