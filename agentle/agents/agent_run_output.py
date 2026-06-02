"""
Updated AgentRunOutput class with enhanced streaming support.

Key additions:
1. New properties for streaming state detection
2. Better handling of partial vs complete states
3. Streaming-specific convenience methods
"""

from collections.abc import Sequence
import logging
from typing import Any

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.context import Context
from agentle.agents.performance_metrics import PerformanceMetrics
from agentle.generations.models.generation.generation import Generation
from agentle.generations.models.message_parts.tool_execution_suggestion import (
    ToolExecutionSuggestion,
)
from agentle.generations.tools.tool_execution_result import ToolExecutionResult

logger = logging.getLogger(__name__)


class AgentRunOutput[T_StructuredOutput](BaseModel):
    """
    Represents the complete result of an agent execution.

    Enhanced to support streaming scenarios where results are delivered incrementally.
    In streaming mode, multiple AgentRunOutput instances are yielded, with the final
    one containing complete information.

    Attributes:
        generation (Generation[T_StructuredOutput] | None): The primary generation produced by the agent.
            In streaming mode, this contains incremental content until the final chunk.

        context (Context): The complete conversation context at the time of this output.
            In streaming mode, this may be a snapshot for intermediate chunks.

        parsed (T_StructuredOutput | None): The structured data extracted from the agent's
            response. Only available in the final chunk for streaming scenarios.

        is_suspended (bool): Whether the execution is suspended and waiting for external input.

        suspension_reason (str | None): The reason why execution was suspended, if applicable.

        resumption_token (str | None): A token that can be used to resume suspended execution.

        performance_metrics (PerformanceMetrics | None): Performance metrics for this execution.
            In streaming mode, these are partial metrics until the final chunk.

        is_streaming_chunk (bool): Whether this is an intermediate streaming chunk.

        is_final_chunk (bool): Whether this is the final chunk in a streaming response.

    Example:
        ```python
        # Non-streaming usage (existing behavior)
        result = await agent.run_async("Tell me about Paris")
        print(result.text)

        # Streaming usage (new behavior)
        async for chunk in agent.run_async("Tell me about Paris", streaming=True):
            print(chunk.text, end="", flush=True)

            if chunk.is_final_chunk:
                print(f"\nFinal tokens: {chunk.generation.usage.total_tokens}")
                print(f"Total time: {chunk.performance_metrics.total_execution_time_ms}ms")
        ```
    """

    generation: Generation[T_StructuredOutput] | None = Field(default=None)
    """
    The generation produced by the agent.
    In streaming mode, contains incremental content until the final chunk.
    """

    context: Context
    """
    The conversation context at the time of this output.
    In streaming mode, may be a snapshot for intermediate chunks.
    """

    parsed: T_StructuredOutput
    """
    Structured data extracted from the agent's response.
    In streaming mode, contains incrementally parsed partial data in each chunk,
    with complete data available in the final chunk.
    """

    generation_text: str = Field(default="")
    """
    The text response from the agent.
    Returns empty string if execution is suspended or generation is None.
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

    suspension_kind: str | None = Field(default=None)
    """
    Machine-readable suspension kind. For example: "tool_approval".
    """

    suspension_data: dict[str, Any] | None = Field(default=None)
    """
    Structured suspension payload for application-level workflows.
    """

    performance_metrics: PerformanceMetrics | None = Field(default=None)
    """
    Performance metrics for this execution.
    In streaming mode, these are partial metrics until the final chunk.
    """

    is_streaming_chunk: bool = Field(default=False)
    """
    Whether this is an intermediate chunk from a streaming response.
    True for all chunks except the final one in streaming mode.
    """

    is_final_chunk: bool = Field(default=True)
    """
    Whether this is the final chunk in a streaming response.
    False for intermediate chunks, True for the final chunk and non-streaming responses.
    """

    @property
    def safe_generation(self) -> Generation[T_StructuredOutput]:
        """Validates if Generation is null"""
        if self.generation is None:
            raise ValueError("Generation is null.")
        return self.generation

    @property
    def tool_execution_results(self) -> Sequence[ToolExecutionResult]:
        return self.context.tool_execution_results

    @property
    def tool_execution_suggestions(self) -> Sequence[ToolExecutionSuggestion]:
        return self.context.tool_execution_suggestions

    @property
    def text(self) -> str:
        """
        The text response from the agent.
        Returns empty string if execution is suspended or generation is None.
        """
        if self.generation is None:
            return ""
        return self.generation.text

    @property
    def is_completed(self) -> bool:
        """
        Whether the execution has completed successfully.
        In streaming mode, only True for the final chunk.
        """
        return (
            not self.is_suspended
            and self.generation is not None
            and self.is_final_chunk
            and self.context.execution_state.state == "completed"
        )

    @property
    def is_streaming(self) -> bool:
        """
        Whether this output is part of a streaming response.
        True if either is_streaming_chunk or not is_final_chunk.
        """
        return self.is_streaming_chunk or not self.is_final_chunk

    @property
    def can_resume(self) -> bool:
        """
        Whether this suspended execution can be resumed.
        """
        return self.is_suspended and self.resumption_token is not None

    @property
    def has_partial_content(self) -> bool:
        """
        Whether this output contains partial content (streaming scenario).
        """
        return self.is_streaming and not self.is_final_chunk

    @property
    def output_tokens(self) -> int:
        """
        Number of tokens in this specific chunk.
        For non-streaming, returns total tokens.
        """
        if self.generation is None:
            return 0
        return self.generation.usage.completion_tokens

    @property
    def input_tokens(self) -> int:
        if self.generation is None:
            return 0
        return self.generation.usage.prompt_tokens

    @property
    def total_tokens_so_far(self) -> int:
        """
        Total tokens processed up to this point in the execution.
        Useful for tracking progress in streaming scenarios.
        """
        if self.performance_metrics is None:
            return 0
        return self.performance_metrics.total_tokens_processed

    def get_streaming_progress(self) -> dict[str, Any]:
        """
        Get progress information for streaming scenarios.

        Returns:
            dict: Progress information including tokens, iterations, and timing.
        """
        if self.performance_metrics is None:
            return {
                "tokens_processed": 0,
                "iterations_completed": 0,
                "execution_time_ms": 0,
                "is_final": self.is_final_chunk,
                "has_tools": False,
            }

        return {
            "tokens_processed": self.performance_metrics.total_tokens_processed,
            "iterations_completed": self.performance_metrics.iteration_count,
            "tool_calls_count": self.performance_metrics.tool_calls_count,
            "execution_time_ms": self.performance_metrics.total_execution_time_ms,
            "generation_time_ms": self.performance_metrics.generation_time_ms,
            "tool_execution_time_ms": self.performance_metrics.tool_execution_time_ms,
            "is_final": self.is_final_chunk,
            "has_tools": self.performance_metrics.tool_calls_count > 0,
            "cache_hit_rate": self.performance_metrics.cache_hit_rate,
        }

    def pretty_formatted(self) -> str:
        """
        Returns a pretty formatted string representation of the AgentRunOutput.
        Enhanced with streaming-specific information.
        """
        lines: list[str] = []
        lines.append("=" * 80)
        lines.append("AGENT RUN OUTPUT")
        if self.is_streaming:
            chunk_type = "FINAL CHUNK" if self.is_final_chunk else "STREAMING CHUNK"
            lines.append(f"({chunk_type})")
        lines.append("=" * 80)

        # Execution Status
        lines.append("\n📊 EXECUTION STATUS:")
        lines.append(f"   • Completed: {self.is_completed}")
        lines.append(f"   • Suspended: {self.is_suspended}")
        lines.append(f"   • Streaming: {self.is_streaming}")
        if self.is_streaming:
            lines.append(f"   • Final Chunk: {self.is_final_chunk}")
            lines.append(f"   • Has Partial Content: {self.has_partial_content}")
        lines.append(f"   • Can Resume: {self.can_resume}")

        # Streaming Progress (if applicable)
        if self.is_streaming:
            lines.append("\n🔄 STREAMING PROGRESS:")
            progress = self.get_streaming_progress()
            lines.append(f"   • Tokens So Far: {progress['tokens_processed']}")
            lines.append(f"   • Iterations: {progress['iterations_completed']}")
            lines.append(f"   • Tool Calls: {progress['tool_calls_count']}")
            lines.append(f"   • Execution Time: {progress['execution_time_ms']:.2f}ms")
            if progress["has_tools"]:
                lines.append(
                    f"   • Generation Time: {progress['generation_time_ms']:.2f}ms"
                )
                lines.append(
                    f"   • Tool Time: {progress['tool_execution_time_ms']:.2f}ms"
                )
            lines.append(f"   • Cache Hit Rate: {progress['cache_hit_rate']:.1f}%")

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

            # Show streaming-specific text info
            if self.is_streaming:
                lines.append(f"   • Chunk Tokens: {self.output_tokens}")
                if not self.is_final_chunk:
                    lines.append("   • Text Preview (chunk):")
                else:
                    lines.append("   • Final Text:")
            else:
                lines.append("   • Text Preview:")

            preview_text = self.generation.text[:100]
            lines.append(
                f"     {preview_text}{'...' if len(self.generation.text) > 100 else ''}"
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
            if self.is_streaming and not self.is_final_chunk:
                lines.append(
                    f"   • Partial Content: {self.text[:200]}{'...' if len(self.text) > 200 else ''}"
                )
            else:
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
            status = (
                "Not available (streaming)"
                if self.is_streaming and not self.is_final_chunk
                else "No"
            )
            lines.append(f"   • Has Parsed Data: {status}")

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
