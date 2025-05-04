from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

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


class AgentToBlackSheepApplicationAdapter(Adapter[Agent[Any], "Application"]):
    extra_routes: Sequence[type[Controller]] = Field(default_factory=list)

    def __init__(self, *extra_routes: type[Controller]):
        self.extra_routes = list(extra_routes)

    def adapt(self, _f: Agent[Any] | A2AInterface[Any]) -> Application:
        """
        Creates a BlackSheep ASGI server for the agent.
        """
        if isinstance(_f, Agent):
            return self._adapt_agent(_f)

        return self._adapt_a2a_interface(_f)

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
