"""
Meta WhatsApp Business API provider for Agentle.

This module provides integration with Meta's official WhatsApp Business API,
offering enterprise-grade WhatsApp messaging capabilities.

Example usage:
    ```python
    from agentle.agents.whatsapp.providers.meta import MetaWhatsAppProvider, MetaWhatsAppConfig

    config = MetaWhatsAppConfig(
        access_token="your_access_token",
        phone_number_id="your_phone_number_id",
        business_account_id="your_business_account_id",
        app_id="your_app_id",
        app_secret="your_app_secret",
        webhook_verify_token="your_webhook_verify_token"
    )

    provider = MetaWhatsAppProvider(config)
    await provider.initialize()
    ```
"""

from agentle.agents.whatsapp.providers.meta.meta_whatsapp_config import (
    MetaWhatsAppConfig,
)
from agentle.agents.whatsapp.providers.meta.meta_whatsapp_provider import (
    MetaWhatsAppProvider,
    MetaWhatsAppError,
)

__all__ = [
    "MetaWhatsAppProvider",
    "MetaWhatsAppConfig",
    "MetaWhatsAppError",
]
