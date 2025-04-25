from rsb.models.base_model import BaseModel

from agentle.agents.agent import Agent
from agentle.agents.models.json_rpc_response import JSONRPCResponse
from agentle.agents.tasks.task import Task
from agentle.agents.tasks.task_get_result import TaskGetResult
from agentle.agents.tasks.task_query_params import TaskQueryParams
from agentle.agents.tasks.task_send_params import TaskSendParams

type WithoutStructuredOutput = None


class TaskManager(BaseModel):
    def send[T_Schema = WithoutStructuredOutput](
        self, task: TaskSendParams, agent: Agent[T_Schema]
    ) -> Task: ...

    def get[T_Schema = WithoutStructuredOutput](
        self, query_params: TaskQueryParams, agent: Agent[T_Schema]
    ) -> TaskGetResult: ...

    def send_subscribe[T_Schema = WithoutStructuredOutput](
        self, task: TaskSendParams, agent: Agent[T_Schema]
    ) -> JSONRPCResponse: ...
