from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any
import json
from pathlib import Path
import urllib.request
import urllib.error

from blacksheep.server.application import Application
from rsb.adapters.adapter import Adapter
from rsb.models.field import Field

from agentle.agents.a2a.a2a_interface import A2AInterface
from agentle.agents.agent import Agent
from agentle.agents.asgi.blacksheep.agent_to_blacksheep_route_handler_adapter import (
    AgentToBlackSheepRouteHandlerAdapter,
)

if TYPE_CHECKING:
    from blacksheep import Application
    from blacksheep.server.controllers import Controller


class AgentToBlackSheepApplicationAdapter(
    Adapter[Agent[Any] | A2AInterface[Any] | str, "Application"]
):
    extra_routes: Sequence[type[Controller]] = Field(default_factory=list)

    def __init__(self, *extra_routes: type[Controller]):
        self.extra_routes = list(extra_routes)

    def adapt(self, _f: Agent[Any] | A2AInterface[Any] | str) -> Application:
        """
        Creates a BlackSheep ASGI server for the agent or A2A interface.

        Args:
            _f: Can be one of:
                - An Agent instance
                - An A2AInterface instance
                - A string path to an agent card JSON file
                - A string URL to an agent card JSON
                - A raw JSON string representing an agent card

        Returns:
            A BlackSheep Application configured to serve the agent or A2A interface.
        """
        # Handle string input (agent card path, URL, or raw JSON)
        if isinstance(_f, str):
            agent = self._load_agent_from_card(_f)
            return self._adapt_agent(agent)

        # Handle Agent or A2AInterface as before
        if isinstance(_f, Agent):
            return self._adapt_agent(_f)

        return self._adapt_a2a_interface(_f)

    def _load_agent_from_card(self, source: str) -> Agent[Any]:
        """
        Loads an agent from an agent card specified in various formats.

        Args:
            source: Can be:
                - A file path to an agent card JSON file
                - A URL to an agent card JSON
                - A raw JSON string representing an agent card

        Returns:
            An Agent instance created from the agent card.

        Raises:
            ValueError: If the agent card cannot be loaded or is invalid.
        """
        agent_card_data = None

        # Check if it's a valid file path
        path = Path(source)
        if path.exists() and path.is_file():
            try:
                with open(path, "r") as f:
                    agent_card_data = json.load(f)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid JSON in agent card file: {source}")

        # Check if it's a URL
        elif source.startswith(("http://", "https://")):
            try:
                with urllib.request.urlopen(source) as response:
                    agent_card_data = json.loads(response.read())
            except (urllib.error.URLError, json.JSONDecodeError) as e:
                raise ValueError(
                    f"Failed to load agent card from URL {source}: {str(e)}"
                )

        # Try parsing as raw JSON
        else:
            try:
                agent_card_data = json.loads(source)
            except json.JSONDecodeError:
                raise ValueError(
                    f"Could not parse input as a file path, URL, or valid JSON: {source}"
                )

        # Create agent from the card data
        try:
            return Agent.from_agent_card(agent_card_data)
        except Exception as e:
            raise ValueError(f"Failed to create agent from agent card: {str(e)}")

    def _adapt_a2a_interface(self, _f: A2AInterface[Any]) -> Application:
        """
        Creates a BlackSheep ASGI application for the A2A interface.

        This creates routes for task management and push notifications.
        """
        import blacksheep
        from blacksheep.server.openapi.ui import ScalarUIProvider
        from blacksheep.server.openapi.v3 import OpenAPIHandler
        from openapidocs.v3 import Info

        app = blacksheep.Application()

        # Get agent name safely
        agent_name = getattr(_f.agent, "name", "Agent")

        # Initialize docs with proper title and description
        docs = OpenAPIHandler(
            ui_path="/openapi",
            info=Info(
                title=f"{agent_name} A2A Interface",
                version="1.0.0",
                description=(
                    f"A2A Interface for {agent_name}. "
                    "This API exposes task management and push notification capabilities."
                ),
            ),
        )
        docs.ui_providers.append(ScalarUIProvider(ui_path="/docs"))
        docs.bind_app(app)

        # Add routes for A2A interface
        controllers = [AgentToBlackSheepRouteHandlerAdapter().adapt(_f)] + list(
            self.extra_routes or []
        )

        app.register_controllers(controllers)

        return app

    def _adapt_agent(self, _f: Agent[Any]) -> Application:
        import blacksheep
        from blacksheep.server.openapi.ui import ScalarUIProvider
        from blacksheep.server.openapi.v3 import OpenAPIHandler
        from openapidocs.v3 import Info

        app = blacksheep.Application()

        docs = OpenAPIHandler(
            ui_path="/openapi",
            info=Info(title=_f.name, version="1.0.0", description=_f.description),
        )
        docs.ui_providers.append(ScalarUIProvider(ui_path="/docs"))
        docs.bind_app(app)

        controllers = [AgentToBlackSheepRouteHandlerAdapter().adapt(_f)] + list(
            self.extra_routes or []
        )

        app.register_controllers(controllers)

        return app
