from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class MetaWhatsAppConfig(BaseModel):
    """Configuration for Meta WhatsApp Business API."""

    access_token: str = Field(description="Meta WhatsApp Business API access token")
    phone_number_id: str = Field(description="WhatsApp Business phone number ID")
    business_account_id: str = Field(description="WhatsApp Business account ID")
    app_id: str = Field(description="Meta app ID")
    app_secret: str = Field(description="Meta app secret")
    webhook_verify_token: str = Field(description="Token for webhook verification")
    webhook_url: str | None = Field(
        default=None, description="Webhook URL for receiving messages"
    )
    api_version: str = Field(
        default="v18.0", description="WhatsApp Business API version"
    )
    base_url: str = Field(
        default="https://graph.facebook.com", description="Base URL for Meta Graph API"
    )
    timeout: int = Field(default=30, description="Request timeout in seconds")
    max_retries: int = Field(
        default=3, description="Maximum number of retries for failed requests"
    )
    retry_delay: float = Field(
        default=1.0, description="Delay between retries in seconds"
    )
