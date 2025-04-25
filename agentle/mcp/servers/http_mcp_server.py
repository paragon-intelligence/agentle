import logging
from collections.abc import Sequence
from contextlib import asynccontextmanager

import httpx
from mcp.types import (
    BlobResourceContents,
    CallToolResult,
    Resource,
    TextResourceContents,
    Tool,
)
from rsb.models.any_url import AnyUrl
from rsb.models.field import Field

from agentle.mcp.servers.mcp_server_protocol import MCPServerProtocol


class HTTPMCPServer(MCPServerProtocol):
    server_name: str = Field(...)
    server_url: AnyUrl = Field(...)
    headers: dict[str, str] = Field(default_factory=dict)
    timeout_in_seconds: float = Field(default=100.0)
    _client: httpx.AsyncClient | None = None
    _logger: logging.Logger = Field(default_factory=lambda: logging.getLogger(__name__))

    async def connect(self) -> None:
        """Connect to the server. For example, this might mean spawning a subprocess or
        opening a network connection. The server is expected to remain connected until
        `cleanup()` is called.
        """
        self._logger.info(f"Conectando ao servidor HTTP: {self.server_url}")
        self._client = httpx.AsyncClient(base_url=str(self.server_url), timeout=30.0)

        # Verificar conexão com o servidor
        try:
            response = await self._client.get("/")
            if response.status_code != 200:
                self._logger.warning(
                    f"Servidor respondeu com status {response.status_code}"
                )
        except Exception as e:
            self._logger.error(f"Erro ao conectar com servidor: {e}")
            await self.cleanup()
            raise ConnectionError(
                f"Não foi possível conectar ao servidor {self.server_url}: {e}"
            )

    @property
    def name(self) -> str:
        """A readable name for the server."""
        return self.server_name

    async def cleanup(self) -> None:
        """Cleanup the server. For example, this might mean closing a subprocess or
        closing a network connection.
        """
        if self._client is not None:
            self._logger.info(f"Fechando conexão com servidor HTTP: {self.server_url}")
            await self._client.aclose()
            self._client = None

    @asynccontextmanager
    async def ensure_connection(self):
        """Context manager to ensure connection is established before operations."""
        try:
            yield
        except httpx.RequestError as e:
            self._logger.error(f"Erro na requisição HTTP: {e}")
            raise

    async def list_tools(self) -> Sequence[Tool]:
        """List the tools available on the server."""
        if self._client is None:
            raise ConnectionError("Servidor não conectado")

        async with self.ensure_connection():
            response = await self._client.get("/tools")
            response.raise_for_status()
            return [Tool.model_validate(tool) for tool in response.json()]

    async def list_resources(self) -> Sequence[Resource]:
        if self._client is None:
            raise ConnectionError("Servidor não conectado")

        async with self.ensure_connection():
            response = await self._client.get("/resources/read")
            response.raise_for_status()
            return [Resource.model_validate(resource) for resource in response.json()]

    async def list_resource_contents(
        self, uri: str
    ) -> Sequence[TextResourceContents | BlobResourceContents]:
        if self._client is None:
            raise ConnectionError("Servidor não conectado")

        async with self.ensure_connection():
            response = await self._client.get(f"/resources/{uri}/contents")
            response.raise_for_status()
            return [
                TextResourceContents.model_validate(content)
                if content["type"] == "text"
                else BlobResourceContents.model_validate(content)
                for content in response.json()
            ]

    async def call_tool(
        self, tool_name: str, arguments: dict[str, object] | None
    ) -> CallToolResult:
        """Invoke a tool on the server."""
        if self._client is None:
            raise ConnectionError("Servidor não conectado")

        async with self.ensure_connection():
            payload = {"tool_name": tool_name, "arguments": arguments or {}}
            response = await self._client.post("/tools/call", json=payload)
            response.raise_for_status()
            return CallToolResult.model_validate(response.json())
