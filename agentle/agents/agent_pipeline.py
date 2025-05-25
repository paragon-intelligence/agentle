"""
Agent Pipeline Module for the Agentle Framework

This module provides functionality for creating sequential pipelines of AI agents,
where the output of one agent becomes the input to the next. Pipelines offer a deterministic
way to decompose complex tasks into a series of simpler steps handled by specialized agents.

Unlike the AgentTeam which uses an orchestrator to dynamically select agents,
AgentPipeline follows a fixed sequence of predefined agents, providing a more
predictable execution flow. This is particularly useful for workflows where:

1. The sequence of operations is known in advance
2. Each step builds upon the results of the previous step
3. A clear division of responsibility between agents is needed

Example:
```python
from agentle.agents.agent import Agent
from agentle.agents.pipelines.agent_pipeline import AgentPipeline
from agentle.generations.providers.google.google_genai_generation_provider import GoogleGenaiGenerationProvider

# Create specialized agents
research_agent = Agent(
    name="Research Agent",
    instructions="You are a research agent. Your task is to gather information on a topic.",
    model="gemini-2.0-flash",
    generation_provider=GoogleGenaiGenerationProvider(),
)

analysis_agent = Agent(
    name="Analysis Agent",
    instructions="You are an analysis agent. Your task is to analyze information and identify patterns.",
    model="gemini-2.0-flash",
    generation_provider=GoogleGenaiGenerationProvider(),
)

summary_agent = Agent(
    name="Summary Agent",
    instructions="You are a summary agent. Your task is to create concise summaries.",
    model="gemini-2.0-flash",
    generation_provider=GoogleGenaiGenerationProvider(),
)

# Create a pipeline of agents
pipeline = AgentPipeline(agents=[research_agent, analysis_agent, summary_agent])

# Run the pipeline
result = pipeline.run("Tell me about renewable energy technologies")
print(result.generation.text)
```
"""

from collections.abc import MutableSequence, Sequence
import logging
from typing import Any, Optional

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.agent import Agent
from agentle.agents.agent_input import AgentInput
from agentle.agents.agent_run_output import AgentRunOutput
from rsb.coroutines.run_sync import run_sync


logger = logging.getLogger(__name__)


class AgentPipeline(BaseModel):
    """
    A sequential pipeline of AI agents where the output of one agent becomes the input to the next.

    AgentPipeline provides a way to chain multiple agents together in a fixed sequence,
    allowing complex tasks to be broken down into manageable steps. Each agent in the pipeline
    specializes in a specific subtask, and the agents work together by passing information
    from one to the next.

    Key features:
    - Sequential execution: Agents run in the exact order specified
    - Automatic input/output handling: Output from one agent becomes input to the next
    - Deterministic workflow: The same pipeline with the same initial input produces consistent results
    - Debug capabilities: Optional logging of intermediate steps for troubleshooting

    Attributes:
        agents: A sequence of Agent instances that form the pipeline.
        debug_mode: When True, enables detailed logging of pipeline execution.

    Examples:
        ```python
        # Create a simple translation pipeline
        english_to_french = Agent(
            name="English to French",
            instructions="Translate English text to French.",
            model="gemini-2.0-flash"
        )

        french_to_spanish = Agent(
            name="French to Spanish",
            instructions="Translate French text to Spanish.",
            model="gemini-2.0-flash"
        )

        # Chain the agents into a pipeline
        translation_pipeline = AgentPipeline(
            agents=[english_to_french, french_to_spanish]
        )

        # Run the pipeline
        result = translation_pipeline.run("Hello, how are you?")
        # Result will be "Hola, ¿cómo estás?" after passing through both agents
        ```

        Creating a complex analysis pipeline:
        ```python
        # Data extraction agent
        extractor = Agent(name="Extractor", instructions="Extract key data points...")

        # Analysis agent
        analyzer = Agent(name="Analyzer", instructions="Analyze the following data...")

        # Recommendation agent
        recommender = Agent(name="Recommender", instructions="Based on the analysis...")

        # Create analysis pipeline
        analysis_pipeline = AgentPipeline(
            agents=[extractor, analyzer, recommender],
            debug_mode=True  # Enable logging for debugging
        )

        # Run pipeline with initial data
        result = analysis_pipeline.run("Raw data: ...")
        ```
    """

    agents: Sequence[Agent[Any]] = Field(
        ...,  # This is a required field
        description="A sequence of Agent instances that will be executed in order.",
    )

    debug_mode: bool = Field(
        default=False,
        description="When True, enables detailed logging of pipeline execution steps.",
    )

    async def resume_async(
        self, resumption_token: str, approval_data: dict[str, Any] | None = None
    ) -> AgentRunOutput[Any]:
        """
        Resume a suspended pipeline execution.

        This method resumes a pipeline that was suspended due to a tool requiring
        human approval. It retrieves the pipeline state and continues execution
        from where it left off.

        Args:
            resumption_token: Token from a suspended pipeline execution
            approval_data: Optional approval data to pass to the resumed execution

        Returns:
            AgentRunOutput with the completed or newly suspended execution

        Raises:
            ValueError: If the resumption token is invalid or not for a pipeline
        """
        if not self.agents:
            raise ValueError("Pipeline must contain at least one agent")

        # Resume the agent execution first
        # We need to get the suspended agent and resume it
        # For now, we'll delegate to the first agent's resume method
        # In a more sophisticated implementation, we'd track which specific agent was suspended
        resumed_result = await self.agents[0].resume_async(
            resumption_token, approval_data
        )

        # Check if we have pipeline state to continue from
        pipeline_state = resumed_result.context.get_checkpoint_data("pipeline_state")
        if not pipeline_state:
            # If no pipeline state, just return the resumed result
            return resumed_result

        # Extract pipeline state
        current_step = pipeline_state.get("current_step", 0)
        current_input = pipeline_state.get("current_input", "")
        intermediate_outputs_data = pipeline_state.get("intermediate_outputs", [])
        debug_mode = pipeline_state.get("debug_mode", self.debug_mode)

        # Reconstruct intermediate outputs for debugging
        intermediate_outputs: MutableSequence[tuple[str, AgentRunOutput[Any]]] = []

        # Continue from the next step after the suspended one
        next_step = current_step + 1

        # Use the resumed result as the current input for the next step
        if resumed_result.generation and resumed_result.generation.text:
            current_input = resumed_result.generation.text

        last_output = resumed_result

        # Continue processing remaining agents
        for i in range(next_step, len(self.agents)):
            agent = self.agents[i]

            if debug_mode:
                logger.info(
                    f"Pipeline resume step {i + 1}/{len(self.agents)}: Running agent '{agent.name}'"
                )

            # Run the current agent with the current input
            output = await agent.run_async(current_input)
            last_output = output

            # Check if this agent also gets suspended
            if output.is_suspended:
                if debug_mode:
                    suspension_msg = (
                        f"Pipeline suspended again at step {i + 1}/{len(self.agents)} "
                        f"(Agent '{agent.name}'): {output.suspension_reason}"
                    )
                    logger.info(suspension_msg)

                # Update pipeline state for the new suspension point
                if output.context:
                    output.context.set_checkpoint_data(
                        "pipeline_state",
                        {
                            "current_step": i,
                            "total_steps": len(self.agents),
                            "current_input": current_input,
                            "intermediate_outputs": intermediate_outputs_data
                            + [
                                {
                                    "agent_name": agent.name,
                                    "output_text": resumed_result.generation.text
                                    if resumed_result.generation
                                    else "",
                                }
                            ],
                            "debug_mode": debug_mode,
                        },
                    )

                return output

            # Store intermediate output if in debug mode
            if debug_mode:
                intermediate_outputs.append((agent.name, output))

            # If this is not the last agent, prepare input for the next agent
            if i < len(self.agents) - 1:
                if output.generation and output.generation.text:
                    current_input = output.generation.text
                else:
                    if debug_mode:
                        logger.warning(
                            f"Agent '{agent.name}' produced no text output. Pipeline terminated early."
                        )
                    break

        return last_output

    def resume(
        self, resumption_token: str, approval_data: dict[str, Any] | None = None
    ) -> AgentRunOutput[Any]:
        """
        Resume a suspended pipeline execution synchronously.

        Args:
            resumption_token: Token from a suspended pipeline execution
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
        Run the agent pipeline synchronously with the provided input.

        This method is a synchronous wrapper around run_async that allows
        the pipeline to be used in synchronous contexts.

        Args:
            input: The initial input to the first agent in the pipeline.
                  Can be a string, UserMessage, Context, or any supported AgentInput type.

        Returns:
            AgentRunOutput: The output from the final agent in the pipeline.

        Example:
            ```python
            pipeline = AgentPipeline(agents=[agent1, agent2, agent3])

            # Simple string input
            result = pipeline.run("Analyze this text")

            # Using a more complex input type
            from agentle.generations.models.messages.user_message import UserMessage
            from agentle.generations.models.message_parts.text import TextPart

            message = UserMessage(parts=[TextPart(text="Analyze this data")])
            result = pipeline.run(message)
            ```
        """
        return run_sync(self.run_async, input=input)

    async def run_async(self, input: AgentInput | Any) -> AgentRunOutput[Any]:
        """
        Executes a pipeline of agents in sequence asynchronously.

        This method processes the agents in the pipeline sequentially, where the output of
        one agent becomes the input to the next agent. The method returns the output of
        the final agent in the pipeline, or the last agent that successfully produced output.

        Args:
            input: The initial input to the first agent in the pipeline

        Returns:
            AgentRunOutput: The result from the final agent in the pipeline

        Raises:
            ValueError: If the pipeline contains no agents
            RuntimeError: If the pipeline execution fails to produce any output

        Example:
            ```python
            pipeline = AgentPipeline(agents=[agent1, agent2, agent3])

            # Using with async/await
            import asyncio

            async def process_data():
                result = await pipeline.run_async("Process this data")
                return result.generation.text

            final_result = asyncio.run(process_data())
            ```

        Notes:
            - If an agent in the middle of the pipeline produces no text output,
              the pipeline will terminate early and return the output of that agent.
            - The debug_mode attribute controls whether detailed logging is performed
              during pipeline execution.
        """
        if not self.agents:
            raise ValueError("Pipeline must contain at least one agent")

        # Store all intermediate outputs if in debug mode
        intermediate_outputs: MutableSequence[tuple[str, AgentRunOutput[Any]]] = []

        # Initialize with the first input
        current_input = input
        last_output: Optional[AgentRunOutput[Any]] = None

        # Process each agent in sequence
        for i, agent in enumerate(self.agents):
            if self.debug_mode:
                logger.info(
                    f"Pipeline step {i + 1}/{len(self.agents)}: Running agent '{agent.name}'"
                )

            # Run the current agent with the current input
            output = await agent.run_async(current_input)
            last_output = output

            # Check if the agent execution was suspended
            if output.is_suspended:
                if self.debug_mode:
                    suspension_msg = (
                        f"Pipeline suspended at step {i + 1}/{len(self.agents)} "
                        f"(Agent '{agent.name}'): {output.suspension_reason}"
                    )
                    logger.info(suspension_msg)

                # Store pipeline state in the context for resumption
                if output.context:
                    output.context.set_checkpoint_data(
                        "pipeline_state",
                        {
                            "current_step": i,
                            "total_steps": len(self.agents),
                            "current_input": current_input,
                            "intermediate_outputs": [
                                {
                                    "agent_name": name,
                                    "output_text": out.generation.text
                                    if out.generation
                                    else "",
                                }
                                for name, out in intermediate_outputs
                            ],
                            "debug_mode": self.debug_mode,
                        },
                    )

                # Return the suspended output immediately
                return output

            # Store intermediate output if in debug mode
            if self.debug_mode:
                intermediate_outputs.append((agent.name, output))

            # If this is not the last agent, prepare input for the next agent
            if i < len(self.agents) - 1:
                # Use the text output as input to the next agent
                if output.generation and output.generation.text:
                    current_input = output.generation.text
                else:
                    # If no text output, we can't continue the pipeline
                    if self.debug_mode:
                        logger.warning(
                            f"Agent '{agent.name}' produced no text output. Pipeline terminated early."
                        )
                    break

        # Ensure we have an output to return
        if last_output is None:
            raise RuntimeError("Pipeline execution failed to produce any output")

        # Add debug information if enabled
        if self.debug_mode and intermediate_outputs:
            logger.debug(f"Pipeline completed with {len(intermediate_outputs)} steps:")
            for i, (name, output) in enumerate(intermediate_outputs):
                output_text = (
                    output.generation.text
                    if output.generation and output.generation.text
                    else ""
                )
                logger.debug(
                    f"  Step {i + 1}: Agent '{name}' - "
                    + f"Output: {output_text[:50]}{'...' if len(output_text) > 50 else ''}"
                )

        return last_output
