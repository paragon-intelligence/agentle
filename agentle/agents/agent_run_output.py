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

    def pretty_formatted(self) -> str:
        """
        Returns a pretty formatted string representation of the AgentRunOutput.

        This method provides a comprehensive view of the agent execution result,
        including all attributes, properties, and execution state information.

        Returns:
            str: A formatted string containing all relevant information about the agent run output.
        """
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("AGENT RUN OUTPUT")
        lines.append("=" * 60)

        # Execution Status
        lines.append("\n📊 EXECUTION STATUS:")
        lines.append(f"   • Completed: {self.is_completed}")
        lines.append(f"   • Suspended: {self.is_suspended}")
        lines.append(f"   • Can Resume: {self.can_resume}")

        # Suspension Information
        if self.is_suspended:
            lines.append("\n⏸️  SUSPENSION DETAILS:")
            lines.append(f"   • Reason: {self.suspension_reason or 'Not specified'}")
            lines.append(
                f"   • Resumption Token: {self.resumption_token or 'Not available'}"
            )

        # Generation Information
        lines.append("\n🤖 GENERATION:")
        if self.generation is not None:
            lines.append("   • Has Generation: Yes")
            lines.append(f"   • Text Length: {len(self.generation.text)} characters")
            lines.append(
                f"   • Text Preview: {self.generation.text[:100]}{'...' if len(self.generation.text) > 100 else ''}"
            )

            # Additional generation attributes if available
            model = getattr(self.generation, "model", None)
            if model:
                lines.append(f"   • Model: {model}")

            finish_reason = getattr(self.generation, "finish_reason", None)
            if finish_reason:
                lines.append(f"   • Finish Reason: {finish_reason}")

            usage = getattr(self.generation, "usage", None)
            if usage:
                lines.append(f"   • Usage: {usage}")
        else:
            lines.append("   • Has Generation: No")

        # Text Property
        lines.append("\n📝 TEXT RESPONSE:")
        if self.text:
            lines.append(f"   • Length: {len(self.text)} characters")
            lines.append(
                f"   • Content: {self.text[:200]}{'...' if len(self.text) > 200 else ''}"
            )
        else:
            lines.append("   • Content: (empty)")

        # Parsed/Structured Output
        lines.append("\n🏗️  STRUCTURED OUTPUT:")
        if self.parsed is not None:
            lines.append("   • Has Parsed Data: Yes")
            lines.append("   • Type: {type(self.parsed).__name__}")
            lines.append(
                f"   • Content: {str(self.parsed)[:200]}{'...' if len(str(self.parsed)) > 200 else ''}"
            )
        else:
            lines.append("   • Has Parsed Data: No")

        # Context Information
        lines.append("\n💬 CONTEXT:")
        if self.context:
            lines.append("   • Has Context: Yes")

            steps = getattr(self.context, "steps", None)
            if steps:
                lines.append(f"   • Number of Steps: {len(steps)}")
                lines.append(
                    f"   • Step Types: {[step.step_type for step in steps[:5]]}"
                )
                if len(steps) > 5:
                    lines.append(f"     (showing first 5 of {len(steps)} steps)")
            else:
                lines.append("   • Number of Steps: 0")

            messages = getattr(self.context, "messages", None)
            if messages:
                lines.append(f"   • Number of Messages: {len(messages)}")

            state = getattr(self.context, "state", None)
            if state:
                lines.append(f"   • State: {state}")
        else:
            lines.append("   • Has Context: No")

        lines.append("\n" + "=" * 60)

        return "\n".join(lines)
