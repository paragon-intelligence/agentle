"""
MCP Servers Integration Example (Synchronous Version)

This example demonstrates how to use the Agentle framework with Model Context Protocol (MCP) servers.
It uses synchronous code only and is structured as a simple script.

Note: This example assumes MCP servers are already running elsewhere. You'll need to
substitute the server URLs and commands with your actual server information.
"""

from agentle.agents.agent import Agent
from agentle.generations.providers.google.google_generation_provider import (
    GoogleGenerationProvider,
)
from agentle.mcp.servers.streamable_http_mcp_server import StreamableHTTPMCPServer

http_server = StreamableHTTPMCPServer(
    server_name="Everything MCP",
    server_url="http://localhost:3001",
)


# Connect to SSE server and list tools
http_server.connect()
print(f"\n🔧 Tools from {http_server.name}:")
sse_tools = http_server.list_tools()
for tool in sse_tools:
    print(f"  - {tool.name}: {tool.description}")

# Create agent with MCP servers
agent = Agent(
    name="MCP-Augmented Assistant",
    description="An assistant that can access files and weather information via MCP servers",
    generation_provider=GoogleGenerationProvider(),
    model="gemini-2.0-flash",
    instructions="""You are a helpful assistant with access to external tools
    Use tools only when necessary.""",
    mcp_servers=[http_server],
)

# Use the with_mcp_servers context manager to automatically connect and cleanup
with agent.with_mcp_servers():
    # Example 1: Query that might use the file system tool
    file_response = agent.run("What is 2+2?")

http_server.cleanup()
