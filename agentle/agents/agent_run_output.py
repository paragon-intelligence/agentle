"""
Module for representing and managing agent execution results.

This module provides the AgentRunOutput class which encapsulates all data
produced during an agent's execution cycle. It represents both the final response
and metadata about the execution process, including conversation steps and structured outputs.

Example:
```python
from agentle.agents.agent import Agent
from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider

# Create and run an agent
agent = Agent(
    generation_provider=GoogleGenerationProvider(),
    model="gemini-2.0-flash",
    instructions="You are a helpful assistant."
)

# The result is an AgentRunOutput object
result = agent.run("What is the capital of France?")

# Access different aspects of the result
text_response = result.generation.text
conversation_steps = result.steps
structured_data = result.parsed  # If using a response_schema
```
"""

import logging

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.context import Context
from agentle.generations.models.generation.generation import Generation

logger = logging.getLogger(__name__)


class AgentRunOutput[T_StructuredOutput](BaseModel):
    """
    Represents the complete result of an agent execution.

    AgentRunOutput encapsulates all data produced when an agent is run, including
    the primary generation response, conversation steps, and optionally
    structured output data when a response schema is provided.

    This class is generic over T_StructuredOutput, which represents the optional
    structured data format that can be extracted from the agent's response when
    a response schema is specified.

    For suspended executions (e.g., waiting for human approval), the generation
    field may be None and the context will contain the suspended state information.

    Attributes:
        generation (Generation[T_StructuredOutput] | None): The primary generation produced by the agent,
            containing the response to the user's input. This includes text, potentially images,
            and any other output format supported by the model. Will be None for suspended executions.

        context (Context): The complete conversation context at the end of execution,
            including execution state, steps, and resumption data.

        parsed (T_StructuredOutput | None): The structured data extracted from the agent's
            response when a response schema was provided. This will be None if
            no schema was specified or if execution is suspended.

        is_suspended (bool): Whether the execution is suspended and waiting for external input
            (e.g., human approval). When True, the agent can be resumed later.

        suspension_reason (str | None): The reason why execution was suspended, if applicable.

        resumption_token (str | None): A token that can be used to resume suspended execution.

    Example:
        ```python
        # Basic usage to access the text response
        result = agent.run("Tell me about Paris")

        if result.is_suspended:
            print(f"Execution suspended: {result.suspension_reason}")
            print(f"Resume with token: {result.resumption_token}")

            # Later, resume the execution
            resumed_result = agent.resume(result.resumption_token, approval_data)
        else:
            response_text = result.generation.text
            print(response_text)

        # Examining conversation steps
        for step in result.context.steps:
            print(f"Step type: {step.step_type}")

        # Working with structured output
        from pydantic import BaseModel

        class CityInfo(BaseModel):
            name: str
            country: str
            population: int

        structured_agent = Agent(
            # ... other parameters ...
            response_schema=CityInfo
        )

        result = structured_agent.run("Tell me about Paris")
        if not result.is_suspended and result.parsed:
            print(f"{result.parsed.name} is in {result.parsed.country}")
            print(f"Population: {result.parsed.population}")
        ```
    """

    generation: Generation[T_StructuredOutput] | None = Field(default=None)
    """
    The generation produced by the agent.
    Will be None for suspended executions.
    """

    context: Context
    """
    The complete conversation context at the end of execution.
    """

    parsed: T_StructuredOutput
    """
    Structured data extracted from the agent's response when a response schema was provided.
    Will be None if no schema was specified or if execution is suspended.
    """

    is_suspended: bool = Field(default=False)
    """
    Whether the execution is suspended and waiting for external input.
    """

    suspension_reason: str | None = Field(default=None)
    """
    The reason why execution was suspended, if applicable.
    """

    resumption_token: str | None = Field(default=None)
    """
    A token that can be used to resume suspended execution.
    """

    @property
    def text(self) -> str:
        """
        The text response from the agent.
        Returns empty string if execution is suspended.
        """
        if self.generation is None:
            return ""
        return self.generation.text

    @property
    def is_completed(self) -> bool:
        """
        Whether the execution has completed successfully.
        """
        return not self.is_suspended and self.generation is not None

    @property
    def can_resume(self) -> bool:
        """
        Whether this suspended execution can be resumed.
        """
        return self.is_suspended and self.resumption_token is not None
