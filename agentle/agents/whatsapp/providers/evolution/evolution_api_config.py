from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class EvolutionAPIConfig(BaseModel):
    """Configuration for Evolution API."""

    base_url: str = Field(description="Base URL of Evolution API instance")
    instance_name: str = Field(description="Instance name in Evolution API")
    api_key: str = Field(description="API key for authentication")
    webhook_url: str | None = Field(
        default=None, description="Webhook URL for receiving messages"
    )
    timeout: int = Field(default=30, description="Request timeout in seconds")
