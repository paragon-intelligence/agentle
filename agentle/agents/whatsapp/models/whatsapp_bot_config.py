from rsb.models.base_model import BaseModel
from rsb.models.field import Field


class WhatsAppBotConfig(BaseModel):
    """Configuration for WhatsApp bot behavior."""

    typing_indicator: bool = Field(
        default=True, description="Show typing indicator while processing"
    )
    typing_duration: int = Field(
        default=3, description="Duration to show typing indicator in seconds"
    )
    auto_read_messages: bool = Field(
        default=True, description="Automatically mark messages as read"
    )
    session_timeout_minutes: int = Field(
        default=30, description="Minutes of inactivity before session reset"
    )
    max_message_length: int = Field(
        default=4096, description="Maximum message length (WhatsApp limit)"
    )
    error_message: str = Field(
        default="Sorry, I encountered an error processing your message. Please try again.",
        description="Default error message",
    )
    welcome_message: str | None = Field(
        default=None, description="Message to send on first interaction"
    )

    # Spam protection and rate limiting
    enable_flood_protection: bool = Field(
        default=True, description="Enable flood protection mechanisms"
    )
    max_messages_per_minute: int = Field(
        default=10, description="Maximum messages allowed per minute from a user"
    )
    flood_ban_duration_minutes: int = Field(
        default=5, description="Duration to ban user after flood detection"
    )
    flood_warning_message: str = Field(
        default="⚠️ You're sending messages too quickly. Please wait a moment.",
        description="Message to send when user is rate limited",
    )
    flood_ban_message: str = Field(
        default="🚫 Too many messages detected. Please wait a few minutes before trying again.",
        description="Message to send when user is temporarily banned",
    )

    # Message batching
    enable_message_batching: bool = Field(
        default=True, description="Batch messages received while processing"
    )
    batch_wait_seconds: float = Field(
        default=2.0,
        description="Additional seconds to wait for more messages after processing",
    )
    batch_separator: str = Field(
        default="\n\n---\n\n", description="Separator between batched messages"
    )
