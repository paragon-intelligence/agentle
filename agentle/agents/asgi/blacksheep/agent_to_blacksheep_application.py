from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from blacksheep.server.application import Application
from rsb.adapters.adapter import Adapter
from rsb.models.field import Field

from agentle.agents.agent import Agent
from agentle.agents.asgi.blacksheep.agent_to_blacksheep_route_handler import (
    AgentToBlackSheepRouteHandler,
)

if TYPE_CHECKING:
    from blacksheep import Application
    from blacksheep.server.controllers import Controller


class AgentToBlackSheepApplication(Adapter[Agent[Any], "Application"]):
    extra_routes: Sequence[type[Controller]] = Field(default_factory=list)

    def __init__(self, *extra_routes: type[Controller]):
        self.extra_routes = list(extra_routes)

    def adapt(self, _f: Agent[Any]) -> Application:
        """
        Creates a BlackSheep ASGI server for the agent.
        """

        import blacksheep
        from blacksheep.server.openapi.ui import ScalarUIProvider
        from blacksheep.server.openapi.v3 import OpenAPIHandler
        from openapidocs.v3 import Info

        app = blacksheep.Application()

        docs: OpenAPIHandler = OpenAPIHandler(
            ui_path="/openapi",
            info=Info(title="Cortex", version="0.0.1", description="app"),
        )
        docs.ui_providers.append(ScalarUIProvider(ui_path="/docs"))
        docs.bind_app(app)

        _ = [AgentToBlackSheepRouteHandler().adapt(_f)] + list(self.extra_routes or [])

        return app
