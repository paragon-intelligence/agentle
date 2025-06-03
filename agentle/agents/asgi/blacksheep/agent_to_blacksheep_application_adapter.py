from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Union
import json
from pathlib import Path
import urllib.request
import urllib.error

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
    from agentle.agents.agent_team import AgentTeam
    from agentle.agents.agent_pipeline import AgentPipeline

# Type alias for all supported agent types
AgentLike = Union[Agent[Any], "AgentTeam", "AgentPipeline"]


class AgentToBlackSheepApplicationAdapter(
    Adapter[AgentLike | A2AInterface[Any] | str, "Application"]
):
    """
    Enhanced BlackSheep adapter that supports converting Agents, Agent Teams,
    Agent Pipelines, and A2A interfaces to production-ready REST APIs.

    This adapter provides automatic OpenAPI documentation, structured endpoints,
    and comprehensive error handling for all agent execution patterns.
    """

    extra_routes: Sequence[type[Controller]] = Field(default_factory=list)

    def __init__(self, *extra_routes: type[Controller]):
        """
        Initialize the adapter with optional extra route controllers.

        Args:
            *extra_routes: Additional BlackSheep controllers to include in the application
        """
        self.extra_routes = list(extra_routes)

    def adapt(self, _f: AgentLike | A2AInterface[Any] | str) -> Application:
        """
        Creates a BlackSheep ASGI server for the agent, team, pipeline, or A2A interface.

        Args:
            _f: Can be one of:
                - An Agent instance
                - An AgentTeam instance
                - An AgentPipeline instance
                - An A2AInterface instance
                - A string path to an agent card JSON file
                - A string URL to an agent card JSON
                - A raw JSON string representing an agent card

        Returns:
            A BlackSheep Application configured to serve the agent system.

        Example:
            ```python
            # Single agent API
            agent = Agent(name="Assistant", ...)
            app = AgentToBlackSheepApplicationAdapter().adapt(agent)

            # Team API
            team = AgentTeam(agents=[agent1, agent2], ...)
            app = AgentToBlackSheepApplicationAdapter().adapt(team)

            # Pipeline API
            pipeline = AgentPipeline(agents=[agent1, agent2, agent3])
            app = AgentToBlackSheepApplicationAdapter().adapt(pipeline)

            # From agent card
            app = AgentToBlackSheepApplicationAdapter().adapt("path/to/agent_card.json")
            ```
        """
        # Handle string input (agent card path, URL, or raw JSON)
        if isinstance(_f, str):
            agent = self._load_agent_from_card(_f)
            return self._adapt_agent(agent)

        # Handle different agent types
        if isinstance(_f, Agent):
            return self._adapt_agent(_f)
        elif self._is_agent_team(_f):
            return self._adapt_agent_team(_f)
        elif self._is_agent_pipeline(_f):
            return self._adapt_agent_pipeline(_f)
        elif isinstance(_f, A2AInterface):
            return self._adapt_a2a_interface(_f)
        else:
            raise ValueError(f"Unsupported type for adaptation: {type(_f)}")

    def _is_agent_team(self, obj: Any) -> bool:
        """Check if object is an AgentTeam without importing (to avoid circular imports)."""
        return (
            hasattr(obj, "agents")
            and hasattr(obj, "orchestrator_provider")
            and hasattr(obj, "run_async")
        )

    def _is_agent_pipeline(self, obj: Any) -> bool:
        """Check if object is an AgentPipeline without importing (to avoid circular imports)."""
        return (
            hasattr(obj, "agents")
            and hasattr(obj, "debug_mode")
            and hasattr(obj, "run_async")
            and not hasattr(obj, "orchestrator_provider")
        )

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
        """
        Creates a BlackSheep ASGI application for a single Agent.

        This creates a simple run endpoint for the agent with automatic documentation.
        """
        import blacksheep
        from blacksheep.server.openapi.ui import ScalarUIProvider
        from blacksheep.server.openapi.v3 import OpenAPIHandler
        from openapidocs.v3 import Info

        app = blacksheep.Application()

        docs = OpenAPIHandler(
            ui_path="/openapi",
            info=Info(
                title=f"{_f.name} API",
                version=_f.version,
                description=f"{_f.description}\n\nThis API provides a single endpoint to interact with the {_f.name} agent.",
            ),
        )
        docs.ui_providers.append(ScalarUIProvider(ui_path="/docs"))
        docs.bind_app(app)

        controllers = [AgentToBlackSheepRouteHandlerAdapter().adapt(_f)] + list(
            self.extra_routes or []
        )

        app.register_controllers(controllers)

        return app

    def _adapt_agent_team(self, team: Any) -> Application:
        """
        Creates a BlackSheep ASGI application for an AgentTeam.

        This creates endpoints for team execution with orchestration capabilities.
        """
        import blacksheep
        from blacksheep.server.openapi.ui import ScalarUIProvider
        from blacksheep.server.openapi.v3 import OpenAPIHandler
        from openapidocs.v3 import Info

        app = blacksheep.Application()

        # Get team information
        team_name = f"Agent Team ({len(team.agents)} agents)"
        agent_names = [
            getattr(agent, "name", f"Agent {i + 1}")
            for i, agent in enumerate(team.agents)
        ]

        description = (
            "Dynamic Agent Team API with intelligent orchestration.\n\n"
            + f"This team consists of {len(team.agents)} specialized agents:\n"
            + "\n".join(f"• {name}" for name in agent_names)
            + "\n\nThe orchestrator dynamically selects the most appropriate agent for each task, "
            + "enabling flexible and intelligent task routing. Teams can handle complex workflows "
            + "that require different types of expertise."
        )

        docs = OpenAPIHandler(
            ui_path="/openapi",
            info=Info(
                title=f"{team_name} API", version="1.0.0", description=description
            ),
        )
        docs.ui_providers.append(ScalarUIProvider(ui_path="/docs"))
        docs.bind_app(app)

        controllers = [AgentToBlackSheepRouteHandlerAdapter().adapt(team)] + list(
            self.extra_routes or []
        )

        app.register_controllers(controllers)

        return app

    def _adapt_agent_pipeline(self, pipeline: Any) -> Application:
        """
        Creates a BlackSheep ASGI application for an AgentPipeline.

        This creates endpoints for pipeline execution with sequential processing.
        """
        import blacksheep
        from blacksheep.server.openapi.ui import ScalarUIProvider
        from blacksheep.server.openapi.v3 import OpenAPIHandler
        from openapidocs.v3 import Info

        app = blacksheep.Application()

        # Get pipeline information
        pipeline_name = f"Agent Pipeline ({len(pipeline.agents)} stages)"
        agent_names = [
            getattr(agent, "name", f"Stage {i + 1}")
            for i, agent in enumerate(pipeline.agents)
        ]

        description = (
            "Sequential Agent Pipeline API for multi-stage processing.\n\n"
            + f"This pipeline processes tasks through {len(pipeline.agents)} sequential stages:\n"
            + "\n".join(f"{i + 1}. {name}" for i, name in enumerate(agent_names))
            + "\n\nEach stage builds upon the output of the previous stage, enabling complex "
            + "workflows where tasks are broken down into specialized steps. The pipeline "
            + "ensures deterministic processing with clear stage-by-stage progression."
        )

        docs = OpenAPIHandler(
            ui_path="/openapi",
            info=Info(
                title=f"{pipeline_name} API", version="1.0.0", description=description
            ),
        )
        docs.ui_providers.append(ScalarUIProvider(ui_path="/docs"))
        docs.bind_app(app)

        controllers = [AgentToBlackSheepRouteHandlerAdapter().adapt(pipeline)] + list(
            self.extra_routes or []
        )

        app.register_controllers(controllers)

        return app
