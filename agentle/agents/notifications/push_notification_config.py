# interface PushNotificationConfig {
#   url: string;
#   token?: string; // token unique to this task/session
#   authentication?: {
#     schemes: string[];
#     credentials?: string;
#   };
# }

from rsb.models.base_model import BaseModel
from rsb.models.field import Field

from agentle.agents.models.authentication import Authentication


class PushNotificationConfig(BaseModel):
    url: str
    token: str | None = Field(default=None)
    authentication: Authentication | None = Field(default=None)
