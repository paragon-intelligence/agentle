Model Context Protocol (MCP)
===========================

The Model Context Protocol (MCP) provides a standardized way for AI agents to access external data sources and tools. Agentle's MCP implementation enables your agents to communicate with both local and remote MCP servers, giving them access to real-time data and specialized capabilities.

Overview
--------

MCP servers provide tools and resources to AI agents, allowing them to:

* Access external data sources
* Execute specialized computation
* Retrieve information from APIs and databases
* Interact with file systems and other resources

Agentle implements the Model Context Protocol by providing:

1. A standardized protocol interface (``MCPServerProtocol``)
2. Multiple server implementations for different transport layers
3. Session management for robustness in production environments
4. Seamless integration with Agentle agents

Server Implementations
---------------------

Agentle provides two main MCP server implementations:

StdioMCPServer
^^^^^^^^^^^^^

The ``StdioMCPServer`` launches and communicates with local MCP servers over standard input/output streams:

.. code-block:: python

    from agentle.mcp.servers.stdio_mcp_server import StdioMCPServer
    
    # Create a stdio-based MCP server
    stdio_server = StdioMCPServer(
        server_name="File System MCP",
        command="/path/to/filesystem_mcp_server",
        server_env={"DEBUG": "1"},
        working_dir="/optional/working/directory"
    )

This implementation is ideal for:

* Local tool execution
* File system operations
* Integration with CLI-based tools

StreamableHTTPMCPServer
^^^^^^^^^^^^^^^^^^^^^^

The ``StreamableHTTPMCPServer`` connects to remote HTTP servers with support for Server-Sent Events (SSE):

.. code-block:: python

    from agentle.mcp.servers.streamable_http_mcp_server import StreamableHTTPMCPServer
    from agentle.mcp.session_management import RedisSessionManager

    # Create a session manager for production environments
    session_manager = RedisSessionManager(
        redis_url="redis://localhost:6379/0",
        key_prefix="agentle_mcp:",
        expiration_seconds=3600  # 1 hour cache
    )
    
    # Create an HTTP-based MCP server with Redis session management
    http_server = StreamableHTTPMCPServer(
        server_name="Weather API MCP",
        server_url="http://localhost:3000",
        mcp_endpoint="/mcp",
        session_manager=session_manager
    )

This implementation is suitable for:

* Remote API integrations
* Web services
* Cloud-based tools
* Multi-process production environments

Session Management
----------------

Agentle provides a flexible session management system for MCP servers, particularly important for the ``StreamableHTTPMCPServer`` in production environments.

The session management system includes:

* Abstract ``SessionManager`` interface
* Thread-safe in-memory implementation for development
* Redis-backed implementation for production

InMemorySessionManager
^^^^^^^^^^^^^^^^^^^^

For development or single-process applications:

.. code-block:: python

    from agentle.mcp.session_management import InMemorySessionManager
    
    # Create an in-memory session manager (thread-safe but process-local)
    memory_session = InMemorySessionManager()
    
    # Use with StreamableHTTPMCPServer
    server = StreamableHTTPMCPServer(
        server_name="Development API",
        server_url="http://localhost:3000",
        session_manager=memory_session
    )

RedisSessionManager
^^^^^^^^^^^^^^^^^

For production, multi-process environments:

.. code-block:: python

    from agentle.mcp.session_management import RedisSessionManager
    
    # Create a Redis-backed session manager for cross-process state
    redis_session = RedisSessionManager(
        redis_url="redis://localhost:6379/0",
        key_prefix="my_app_mcp:",
        expiration_seconds=3600  # 1 hour session lifetime
    )
    
    # Use with StreamableHTTPMCPServer
    server = StreamableHTTPMCPServer(
        server_name="Production API",
        server_url="https://api.example.com",
        session_manager=redis_session
    )

Using MCP Servers with Agents
----------------------------

You can use MCP servers with Agentle agents using the ``start_mcp_servers()`` context manager:

.. code-block:: python

    from agentle.agents.agent import Agent
    from agentle.mcp.servers.streamable_http_mcp_server import StreamableHTTPMCPServer
    from agentle.generations.providers.google.google_generation_provider import GoogleGenerationProvider

    # Create an agent with MCP server
    agent = Agent(
        name="MCP-Enhanced Assistant",
        generation_provider=GoogleGenerationProvider(),
        model="gemini-2.5-flash",
        instructions="You're an assistant with access to external tools.",
        mcp_servers=[StreamableHTTPMCPServer(
            server_name="Data API",
            server_url="http://localhost:3000"
        )]
    )

    # Use the context manager to handle connection lifecycle
    with agent.start_mcp_servers():
        # Agent can now use tools from the MCP server
        response = agent.run("What's in the /documents folder?")

For async usage:

.. code-block:: python

    async with agent.start_mcp_servers_async():
        response = await agent.run_async("What's in the /documents folder?")

Creating Custom Session Managers
------------------------------

You can implement custom session managers by extending the ``SessionManager`` abstract base class:

.. code-block:: python

    from typing import Dict, Optional, Any
    from agentle.mcp.session_management import SessionManager
    
    class MyCustomSessionManager(SessionManager):
        """Custom session manager implementation."""
        
        async def get_session(self, server_key: str) -> Optional[Dict[str, Any]]:
            # Implement session retrieval
            ...
        
        async def store_session(self, server_key: str, session_data: Dict[str, Any]) -> None:
            # Implement session storage
            ...
        
        async def delete_session(self, server_key: str) -> None:
            # Implement session deletion
            ...
        
        async def close(self) -> None:
            # Implement resource cleanup
            ...

Best Practices
------------

1. **Choose the right session manager for your deployment**:
   - Use ``InMemorySessionManager`` for development and testing
   - Use ``RedisSessionManager`` for production with multiple servers/workers

2. **Always use context managers** for proper connection lifecycle:
   - ``with agent.start_mcp_servers()`` for synchronous code
   - ``async with agent.start_mcp_servers_async()`` for asynchronous code

3. **Set appropriate timeouts** based on your operations:
   - ``timeout_s`` for HTTP requests
   - ``expiration_seconds`` for Redis sessions

4. **Handle errors gracefully**:
   - Connection errors
   - Session expiration
   - Tool execution failures 