from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from agentle.agents.agent_input import AgentInput
    from agentle.agents.agent_run_output import AgentRunOutput
    from agentle.generations.models.generation.trace_params import TraceParams


@runtime_checkable
class Runnable[T = None](Protocol):
    def run(
        self,
        input: AgentInput | Any,
        *,
        timeout: float | None = None,
        trace_params: TraceParams | None = None,
    ) -> AgentRunOutput[T]: ...

    async def run_async(
        self, input: AgentInput | Any, *, trace_params: TraceParams | None = None
    ) -> AgentRunOutput[T]: ...
