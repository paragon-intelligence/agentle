"""
A2A Authentication Model

This module defines the Authentication class, which represents authentication information
for secure communication in the A2A protocol. It enables secure access to protected
resources and services.
"""

from collections.abc import Sequence
from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class Authentication(BaseModel):
    """
    Represents authentication information for secure communication.

    This class encapsulates authentication details required for secure access to
    protected resources and services in the A2A protocol. It supports various
    authentication schemes such as Basic and Bearer token authentication.

    Attributes:
        schemes: List of supported authentication schemes (e.g., "Basic", "Bearer")
        credentials: Optional credentials string for authentication

    Example:
        ```python
        from agentle.agents.a2a.models.authentication import Authentication

        # Create basic authentication
        basic_auth = Authentication(
            schemes=["Basic"],
            credentials="base64encoded_username_password"
        )

        # Create bearer token authentication
        bearer_auth = Authentication(
            schemes=["Bearer"],
            credentials="jwt_token_string"
        )

        # Create authentication with multiple supported schemes
        multi_scheme_auth = Authentication(
            schemes=["Basic", "Bearer"],
            credentials="appropriate_credentials_string"
        )
        ```

    Note:
        When using Basic authentication, the credentials should be Base64-encoded
        in the format "username:password". For Bearer authentication, the credentials
        should be the raw token string.
    """

    schemes: Sequence[str]
    """
    e.g. Basic, Bearer
    """

    credentials: str | None = Field(default=None)
    """
    Credentials a client should use for private cards
    """
