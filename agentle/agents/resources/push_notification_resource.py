from rsb.models.base_model import BaseModel

from agentle.agents.agent import Agent
from agentle.agents.notifications.push_notification_config import PushNotificationConfig
from agentle.agents.notifications.task_push_notification_config import (
    TaskPushNotificationConfig,
)
from agentle.agents.tasks.task_get_result import TaskGetResult
from agentle.agents.tasks.task_query_params import TaskQueryParams

type WithoutStructuredOutput = None


class PushNotificationResource[T_Schema = WithoutStructuredOutput](BaseModel):
    agent: Agent[T_Schema]

    def set(self, config: PushNotificationConfig) -> TaskPushNotificationConfig: ...

    def get(self, query_params: TaskQueryParams) -> TaskGetResult: ...
