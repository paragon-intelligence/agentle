from rsb.models.base_model import BaseModel

from agentle.agents.agent import Agent


from agentle.agents.models.json_rpc_response import JSONRPCResponse
from agentle.agents.resources.push_notification_resource import PushNotificationResource
from agentle.agents.tasks.task import Task
from agentle.agents.tasks.managment.task_manager import TaskManager
from agentle.agents.tasks.task_get_result import TaskGetResult
from agentle.agents.tasks.task_query_params import TaskQueryParams
from agentle.agents.tasks.task_send_params import TaskSendParams

type WithoutStructuredOutput = None


class TaskResource[T_Schema = WithoutStructuredOutput](BaseModel):
    agent: Agent[T_Schema]
    manager: TaskManager

    @property
    def pushNotification(self) -> PushNotificationResource[T_Schema]:
        return PushNotificationResource(agent=self.agent)

    def send(self, task: TaskSendParams) -> Task:
        return self.manager.send(task, agent=self.agent)

    def get(self, query_params: TaskQueryParams) -> TaskGetResult:
        return self.manager.get(query_params, agent=self.agent)

    def send_subscribe(self, task: TaskSendParams) -> JSONRPCResponse:
        return self.manager.send_subscribe(task, agent=self.agent)
