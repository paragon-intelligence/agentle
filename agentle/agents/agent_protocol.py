from __future__ import annotations

from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from agentle.agents.agent_input import AgentInput
    from agentle.agents.agent_run_output import AgentRunOutput
    from agentle.generations.models.generation.trace_params import TraceParams


@runtime_checkable
class AgentProtocol[T = None](Protocol):
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

    @contextmanager
    def start_mcp_servers(self) -> Generator[None, None, None]:
        """
        Context manager to connect and clean up MCP servers.

        This context manager ensures that all MCP servers are connected before the
        code block is executed and cleaned up after completion, even in case of exceptions.

        Yields:
            None: Does not return a value, just manages the context.

        Example:
            ```python
            async with agent.start_mcp_servers():
                # Operations that require connection to MCP servers
                result = await agent.run_async("Query to server")
            # Servers are automatically disconnected here
            ```
        """
        ...

    @asynccontextmanager
    async def start_mcp_servers_async(self) -> AsyncGenerator[None, None]:
        """
        Asynchronous context manager to connect and clean up MCP servers.

        This method ensures that all MCP servers are connected before the
        code block is executed and cleaned up after completion, even in case of exceptions.

        Yields:
            None: Does not return a value, just manages the context.

        Example:
            ```python
            async with agent.start_mcp_servers():
                # Operations that require connection to MCP servers
                result = await agent.run_async("Query to server")
            # Servers are automatically disconnected here
            ```
        """
        ...
