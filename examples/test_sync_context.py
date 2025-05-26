#!/usr/bin/env python3
"""
Test example to demonstrate that the run_sync fix works in synchronous contexts.
This example uses a mock MCP server that doesn't require external dependencies.
"""

from agentle.agents.agent import Agent
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.models.message_parts.text import TextPart
from agentle.generations.providers.google.google_generation_provider import (
    GoogleGenerationProvider,
)
from agentle.mcp.servers.mcp_server_protocol import MCPServerProtocol
from collections.abc import MutableMapping, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp.types import (
        BlobResourceContents,
        CallToolResult,
        Resource,
        TextResourceContents,
        Tool,
    )


class MockMCPServer(MCPServerProtocol):
    """A mock MCP server for testing that doesn't require external connections."""

    @property
    def name(self) -> str:
        return "MockMCPServer"

    async def connect_async(self) -> None:
        """Mock connection that always succeeds."""
        print(f"✅ {self.name} connected successfully!")

    async def cleanup_async(self) -> None:
        """Mock cleanup."""
        print(f"🧹 {self.name} cleaned up successfully!")

    async def list_tools_async(self) -> Sequence["Tool"]:
        """Return empty list of tools."""
        return []

    async def list_resources_async(self) -> Sequence["Resource"]:
        """Return empty list of resources."""
        return []

    async def list_resource_contents_async(
        self, uri: str
    ) -> Sequence["TextResourceContents | BlobResourceContents"]:
        """Return empty list of resource contents."""
        return []

    async def call_tool_async(
        self, tool_name: str, arguments: MutableMapping[str, object] | None
    ) -> "CallToolResult":
        """Mock tool call."""
        raise NotImplementedError("Mock server doesn't implement tool calls")


def main():
    """Main function to test synchronous context."""
    print("🧪 Testing Agent with MCP servers in synchronous context...")
    print("=" * 60)

    # Create a mock MCP server
    mock_server = MockMCPServer()

    # Create an agent with the mock MCP server
    agent = Agent(
        name="TestAgent",
        description="An agent for testing synchronous context with MCP servers.",
        generation_provider=GoogleGenerationProvider(),
        mcp_servers=[mock_server],
    )

    print("🤖 Agent created successfully!")
    print("🔌 Testing MCP server connection in synchronous context...")

    # This should work now with our run_sync fix
    try:
        with agent.start_mcp_servers():
            print("🎉 MCP servers started successfully in synchronous context!")

            # Create a simple user message
            user_message = UserMessage(parts=[TextPart(text="Hello, test agent!")])

            # Run the agent (this will also test run_sync)
            print("🤔 Running agent...")
            result = agent.run(user_message)

            # Print the response
            print(f"🤖 Agent response: {result.text}")
            print(f"📊 Context steps: {len(result.context.steps)}")

        print("✅ All tests completed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
