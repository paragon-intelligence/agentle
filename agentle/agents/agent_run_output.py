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
        lines.append("=" * 80)
        lines.append("AGENT RUN OUTPUT")
        lines.append("=" * 80)

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

        # Detailed Execution State Information
        if self.context and self.context.execution_state:
            exec_state = self.context.execution_state
            lines.append("\n🔄 EXECUTION STATE:")
            lines.append(f"   • State: {exec_state.state}")
            lines.append(
                f"   • Current Iteration: {exec_state.current_iteration} / {exec_state.max_iterations}"
            )
            lines.append(f"   • Total Tool Calls: {exec_state.total_tool_calls}")
            lines.append(f"   • Resumable: {exec_state.resumable}")

            # Timing Information
            if exec_state.started_at:
                lines.append(
                    f"   • Started At: {exec_state.started_at.strftime('%Y-%m-%d %H:%M:%S')}"
                )
            if exec_state.completed_at:
                lines.append(
                    f"   • Completed At: {exec_state.completed_at.strftime('%Y-%m-%d %H:%M:%S')}"
                )
            if exec_state.paused_at:
                lines.append(
                    f"   • Paused At: {exec_state.paused_at.strftime('%Y-%m-%d %H:%M:%S')}"
                )

            lines.append(
                f"   • Last Updated: {exec_state.last_updated_at.strftime('%Y-%m-%d %H:%M:%S')}"
            )

            if exec_state.total_duration_ms:
                lines.append(
                    f"   • Total Duration: {exec_state.total_duration_ms:.2f}ms"
                )

            if exec_state.error_message:
                lines.append(f"   • Error: {exec_state.error_message}")

            if exec_state.pause_reason:
                lines.append(f"   • Pause Reason: {exec_state.pause_reason}")

            if exec_state.checkpoint_data:
                lines.append(
                    f"   • Checkpoint Data: {len(exec_state.checkpoint_data)} items"
                )

        # Enhanced Generation Information
        lines.append("\n🤖 GENERATION:")
        if self.generation is not None:
            lines.append("   • Has Generation: Yes")
            lines.append(f"   • Generation ID: {self.generation.id}")
            lines.append(
                f"   • Created: {self.generation.created.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            lines.append(f"   • Model: {self.generation.model}")
            lines.append(f"   • Choices: {len(self.generation.choices)}")
            lines.append(f"   • Text Length: {len(self.generation.text)} characters")
            lines.append(
                f"   • Text Preview: {self.generation.text[:100]}{'...' if len(self.generation.text) > 100 else ''}"
            )

            # Usage information from generation
            if hasattr(self.generation, "usage") and self.generation.usage:
                usage = self.generation.usage
                lines.append(
                    f"   • Token Usage: {usage.prompt_tokens} prompt + {usage.completion_tokens} completion = {usage.total_tokens} total"
                )
        else:
            lines.append("   • Has Generation: No")

        # Enhanced Text Response
        lines.append("\n📝 TEXT RESPONSE:")
        if self.text:
            lines.append(f"   • Length: {len(self.text)} characters")
            lines.append(f"   • Word Count: {len(self.text.split())} words")
            lines.append(
                f"   • Content: {self.text[:200]}{'...' if len(self.text) > 200 else ''}"
            )
        else:
            lines.append("   • Content: (empty)")

        # Enhanced Structured Output
        lines.append("\n🏗️  STRUCTURED OUTPUT:")
        if self.parsed is not None:
            lines.append("   • Has Parsed Data: Yes")
            lines.append(f"   • Type: {type(self.parsed).__name__}")
            lines.append(
                f"   • Content: {str(self.parsed)[:200]}{'...' if len(str(self.parsed)) > 200 else ''}"
            )
        else:
            lines.append("   • Has Parsed Data: No")

        # Enhanced Context Information
        lines.append("\n💬 CONTEXT:")
        if self.context:
            lines.append("   • Has Context: Yes")
            lines.append(f"   • Context ID: {self.context.context_id}")

            if self.context.session_id:
                lines.append(f"   • Session ID: {self.context.session_id}")

            if self.context.parent_context_id:
                lines.append(
                    f"   • Parent Context ID: {self.context.parent_context_id}"
                )

            if self.context.tags:
                lines.append(f"   • Tags: {', '.join(self.context.tags)}")

            if self.context.metadata:
                lines.append(f"   • Metadata: {len(self.context.metadata)} items")

            # Message history breakdown
            if self.context.message_history:
                from agentle.generations.models.messages.user_message import UserMessage
                from agentle.generations.models.messages.assistant_message import (
                    AssistantMessage,
                )
                from agentle.generations.models.messages.developer_message import (
                    DeveloperMessage,
                )

                user_count = sum(
                    1
                    for msg in self.context.message_history
                    if isinstance(msg, UserMessage)
                )
                assistant_count = sum(
                    1
                    for msg in self.context.message_history
                    if isinstance(msg, AssistantMessage)
                )
                developer_count = sum(
                    1
                    for msg in self.context.message_history
                    if isinstance(msg, DeveloperMessage)
                )

                lines.append(
                    f"   • Messages: {len(self.context.message_history)} total"
                )
                lines.append(
                    f"     - User: {user_count}, Assistant: {assistant_count}, Developer: {developer_count}"
                )

            # Token usage from context
            if self.context.total_token_usage:
                usage = self.context.total_token_usage
                lines.append(
                    f"   • Total Token Usage: {usage.prompt_tokens} prompt + {usage.completion_tokens} completion = {usage.total_tokens} total"
                )

            # Detailed Steps Information
            if self.context.steps:
                lines.append(f"   • Steps: {len(self.context.steps)} total")

                # Step type breakdown
                step_types = {}
                successful_steps = 0
                failed_steps = 0
                total_step_duration = 0.0

                for step in self.context.steps:
                    step_types[step.step_type] = step_types.get(step.step_type, 0) + 1
                    if step.is_successful:
                        successful_steps += 1
                    else:
                        failed_steps += 1
                    if step.duration_ms:
                        total_step_duration += step.duration_ms

                lines.append(
                    f"     - Successful: {successful_steps}, Failed: {failed_steps}"
                )
                if step_types:
                    step_type_str = ", ".join(
                        [f"{k}: {v}" for k, v in step_types.items()]
                    )
                    lines.append(f"     - Types: {step_type_str}")

                if total_step_duration > 0:
                    lines.append(
                        f"     - Total Step Duration: {total_step_duration:.2f}ms"
                    )

                # Show detailed info for recent steps (last 3)
                lines.append("\n🔍 RECENT STEPS:")
                recent_steps = list(self.context.steps)[-3:]
                for i, step in enumerate(recent_steps, 1):
                    lines.append(
                        f"   Step {len(self.context.steps) - len(recent_steps) + i}:"
                    )
                    lines.append(f"     • Type: {step.step_type}")
                    lines.append(
                        f"     • Timestamp: {step.timestamp.strftime('%H:%M:%S')}"
                    )
                    lines.append(f"     • Iteration: {step.iteration}")
                    lines.append(f"     • Successful: {step.is_successful}")

                    if step.duration_ms:
                        lines.append(f"     • Duration: {step.duration_ms:.2f}ms")

                    if step.tool_execution_suggestions:
                        lines.append(
                            f"     • Tool Calls: {len(step.tool_execution_suggestions)}"
                        )
                        for tool_call in step.tool_execution_suggestions[
                            :2
                        ]:  # Show first 2
                            lines.append(
                                f"       - {tool_call.tool_name}({', '.join(str(k) for k in tool_call.args.keys())})"
                            )

                    if step.tool_execution_results:
                        successful_tools = sum(
                            1
                            for result in step.tool_execution_results
                            if result.success
                        )
                        failed_tools = (
                            len(step.tool_execution_results) - successful_tools
                        )
                        lines.append(
                            f"     • Tool Results: {successful_tools} successful, {failed_tools} failed"
                        )

                    if step.generation_text:
                        preview = step.generation_text[:100]
                        lines.append(
                            f"     • Generated Text: {preview}{'...' if len(step.generation_text) > 100 else ''}"
                        )

                    if step.reasoning:
                        reasoning_preview = step.reasoning[:100]
                        lines.append(
                            f"     • Reasoning: {reasoning_preview}{'...' if len(step.reasoning) > 100 else ''}"
                        )

                    if step.token_usage:
                        lines.append(
                            f"     • Tokens: {step.token_usage.prompt_tokens}+{step.token_usage.completion_tokens}={step.token_usage.total_tokens}"
                        )

                    if step.error_message:
                        lines.append(f"     • Error: {step.error_message}")
            else:
                lines.append("   • Steps: 0")
        else:
            lines.append("   • Has Context: No")

        lines.append("\n" + "=" * 80)

        return "\n".join(lines)

    def __str__(self) -> str:
        return self.pretty_formatted()
