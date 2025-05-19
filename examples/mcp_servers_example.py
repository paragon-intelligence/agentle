"""
MCP Servers Integration Example (Synchronous Version)

This example demonstrates how to use the Agentle framework with Model Context Protocol (MCP) servers.
It uses synchronous code only and is structured as a simple script.

Note: This example assumes MCP servers are already running elsewhere. You'll need to
substitute the server URLs and commands with your actual server information.
"""

import os

from agentle.agents.agent import Agent
from agentle.generations.providers.google.google_genai_generation_provider import (
    GoogleGenaiGenerationProvider,
)
from agentle.mcp.servers.streamable_http_mcp_server import StreamableHTTPMCPServer
from agentle.mcp.servers.stdio_mcp_server import StdioMCPServer

# Set up provider
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Please set GOOGLE_API_KEY environment variable")

provider = GoogleGenaiGenerationProvider(api_key=api_key)

# Create MCP servers
# Note: Replace these with your actual server URLs and commands
stdio_server = StdioMCPServer(
    server_name="File System MCP",
    command="/path/to/filesystem_mcp_server",  # Replace with actual command
    server_env={"DEBUG": "1"},
)

http_server = StreamableHTTPMCPServer(
    server_name="Weather API MCP",
    server_url="http://localhost:3000",  # Replace with actual server URL
)

# List tools from the servers (synchronous wrapper over async methods)
print("\n=== LISTING AVAILABLE TOOLS ===")

# Connect to stdio server and list tools
stdio_server.connect()
print(f"\n🔧 Tools from {stdio_server.name}:")
stdio_tools = stdio_server.list_tools()
for tool in stdio_tools:
    print(f"  - {tool.name}: {tool.description}")

# Connect to SSE server and list tools
http_server.connect()
print(f"\n🔧 Tools from {http_server.name}:")
sse_tools = http_server.list_tools()
for tool in sse_tools:
    print(f"  - {tool.name}: {tool.description}")

# Create agent with MCP servers
print("\n=== CREATING AGENT WITH MCP SERVERS ===")
agent = Agent(
    name="MCP-Augmented Assistant",
    description="An assistant that can access files and weather information via MCP servers",
    generation_provider=provider,
    model="gemini-2.0-flash",
    instructions="""You are a helpful assistant with access to external tools
    through MCP servers. You can access files from the filesystem and
    get weather information for locations.""",
    mcp_servers=[stdio_server, http_server],
)

# Use the start_mcp_servers context manager to automatically connect and cleanup
print("\n=== RUNNING AGENT QUERIES ===")
with agent.start_mcp_servers():
    # Example 1: Query that might use the file system tool
    print("\n--- FILE SYSTEM QUERY ---")
    file_query = "Can you show me the contents of the README.md file?"
    print(f"Query: {file_query}")
    file_response = agent.run(file_query)
    print(f"Response: {file_response.generation.text}")

    # Example 2: Query that might use the weather tool
    print("\n--- WEATHER QUERY ---")
    weather_query = "What's the weather like in Tokyo today?"
    print(f"Query: {weather_query}")
    weather_response = agent.run(weather_query)
    print(f"Response: {weather_response.generation.text}")

    # Example 3: Query that might use both tools
    print("\n--- COMBINED QUERY ---")
    combined_query = "Find weather forecast files in the data directory and summarize the Tokyo forecast."
    print(f"Query: {combined_query}")
    combined_response = agent.run(combined_query)
    print(f"Response: {combined_response.generation.text}")

print("\n=== EXAMPLE COMPLETE ===")

# Clean up remaining connections
stdio_server.cleanup()
http_server.cleanup()
