"""
Example demonstrating how to use the SSEMCPServer for connecting to MCP servers over SSE.

This example shows how to:
1. Create an SSE MCP server connection
2. List available tools and resources
3. Call tools and read resources
4. Handle connection lifecycle properly
"""

import logging
from typing import Any, Dict

from agentle.mcp.servers.sse_mcp_server import SSEMCPServer
from agentle.mcp.session_management import InMemorySessionManager


# Configure logging to see what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """
    Main example function demonstrating SSE MCP Server usage.
    """
    # Create a session manager (use InMemorySessionManager for development)
    session_manager = InMemorySessionManager()

    # Create an SSE MCP server instance
    sse_server = SSEMCPServer(
        server_name="Example SSE MCP Server",
        server_url="http://localhost:3000",  # Replace with your SSE server URL
        sse_endpoint="/sse",  # SSE endpoint for server-to-client streaming
        messages_endpoint="/messages",  # POST endpoint for client-to-server messages
        session_manager=session_manager,
        timeout_s=30.0,  # 30 second timeout
        headers={
            "Authorization": "Bearer your-token-here",  # Add auth if needed
            "User-Agent": "agentle-mcp-client/1.0",
        },
    )

    try:
        # Connect to the SSE server
        logger.info("Connecting to SSE MCP server...")
        sse_server.connect()
        logger.info(f"Connected to server: {sse_server.name}")

        # List available tools
        logger.info("Listing available tools...")
        tools = sse_server.list_tools()
        logger.info(f"Found {len(tools)} tools:")
        for tool in tools:
            logger.info(f"  - {tool.name}: {tool.description}")

        # List available resources
        logger.info("Listing available resources...")
        resources = sse_server.list_resources()
        logger.info(f"Found {len(resources)} resources:")
        for resource in resources:
            logger.info(f"  - {resource.name} ({resource.uri}): {resource.description}")

        # Example: Call a tool (if any tools are available)
        if tools:
            first_tool = tools[0]
            logger.info(f"Calling tool: {first_tool.name}")

            # Example arguments - adjust based on your tool's schema
            tool_arguments: Dict[str, Any] = {}

            try:
                result = sse_server.call_tool(first_tool.name, tool_arguments)
                logger.info(f"Tool result: {result}")
            except Exception as e:
                logger.error(f"Error calling tool {first_tool.name}: {e}")

        # Example: Read a resource (if any resources are available)
        if resources:
            first_resource = resources[0]
            logger.info(f"Reading resource: {first_resource.uri}")

            try:
                contents = sse_server.list_resource_contents(str(first_resource.uri))
                logger.info(f"Resource contents: {len(contents)} items")
                for content in contents:
                    # Check content type and access appropriate attribute
                    from mcp.types import TextResourceContents, BlobResourceContents

                    if isinstance(content, TextResourceContents) and content.text:
                        logger.info(f"  Text content: {content.text[:100]}...")
                    elif isinstance(content, BlobResourceContents) and content.blob:
                        logger.info(f"  Binary content: {len(content.blob)} bytes")
            except Exception as e:
                logger.error(f"Error reading resource {str(first_resource.uri)}: {e}")

    except Exception as e:
        logger.error(f"Error during SSE MCP server operations: {e}")
    finally:
        # Always clean up the connection
        logger.info("Cleaning up SSE MCP server connection...")
        sse_server.cleanup()
        logger.info("Connection cleaned up successfully")


async def production_example():
    """
    Example showing production-ready configuration with Redis session management.
    """
    from agentle.mcp.session_management import RedisSessionManager

    # Create a Redis session manager for production
    redis_session_manager = RedisSessionManager(
        redis_url="redis://localhost:6379/0",
        key_prefix="agentle_sse_mcp:",
        expiration_seconds=3600,  # 1 hour session timeout
    )

    # Create SSE server with production configuration
    sse_server = SSEMCPServer(
        server_name="Production SSE MCP Server",
        server_url="https://api.example.com",  # Production server URL
        sse_endpoint="/mcp/sse",
        messages_endpoint="/mcp/messages",
        session_manager=redis_session_manager,
        timeout_s=60.0,  # Longer timeout for production
        headers={
            "Authorization": "Bearer production-token",
            "User-Agent": "agentle-mcp-client/1.0",
            "X-Client-Version": "1.0.0",
        },
    )

    try:
        sse_server.connect()
        logger.info("Connected to production SSE MCP server")

        # Your production logic here...
        tools = sse_server.list_tools()
        logger.info(f"Production server has {len(tools)} tools available")

    except Exception as e:
        logger.error(f"Production SSE MCP server error: {e}")
    finally:
        sse_server.cleanup()


if __name__ == "__main__":
    # Run the basic example
    print("Running SSE MCP Server example...")
    main()
