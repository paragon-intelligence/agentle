from __future__ import annotations

from typing import TYPE_CHECKING

from rsb.adapters.adapter import Adapter

from agentle.generations.models.messages.assistant_message import AssistantMessage
from agentle.generations.models.messages.developer_message import DeveloperMessage
from agentle.generations.models.messages.user_message import UserMessage
from agentle.generations.providers.google.function_calling_config import (
    FunctionCallingConfig,
)
from agentle.generations.providers.google.google_generation_provider import (
    GoogleGenerationProvider,
)
from agentle.generations.tracing.contracts.stateful_observability_client import (
    StatefulObservabilityClient,
)

if TYPE_CHECKING:
    from google.auth.credentials import Credentials
    from google.genai.client import (
        DebugConfig,
    )
    from google.genai.types import Content, HttpOptions


class GoogleVertexaiGenerationProvider(GoogleGenerationProvider):
    def __init__(
        self,
        *,
        tracing_client: StatefulObservabilityClient | None = None,
        api_key: str | None | None = None,
        credentials: Credentials | None = None,
        project: str | None = None,
        location: str | None = None,
        debug_config: DebugConfig | None = None,
        http_options: HttpOptions | None = None,
        message_adapter: Adapter[
            AssistantMessage | UserMessage | DeveloperMessage, Content
        ]
        | None = None,
        function_calling_config: FunctionCallingConfig | None = None,
    ) -> None:
        """
        Initialize the Google Generation Provider.

        Args:
            tracing_client: Optional client for observability and tracing.
            api_key: Optional API key for authentication with Google AI.
            credentials: Optional credentials object for authentication.
            project: Google Cloud project ID (required for Vertex AI).
            location: Google Cloud region (required for Vertex AI).
            debug_config: Optional configuration for debug logging.
            http_options: HTTP options for the Google AI client.
            message_adapter: Optional adapter to convert Agentle messages to Google Content.
            function_calling_config: Optional configuration for function calling behavior.
        """
        super().__init__(
            tracing_client=tracing_client,
            use_vertex_ai=True,
            api_key=api_key,
            credentials=credentials,
            project=project,
            location=location,
            debug_config=debug_config,
            http_options=http_options,
            message_adapter=message_adapter,
            function_calling_config=function_calling_config,
        )
