from rsb.models.base_model import BaseModel

from agentle.agents.notifications.push_notification_config import PushNotificationConfig


class TaskPushNotificationConfig(BaseModel):
    id: str
    pushNotificationConfig: PushNotificationConfig
