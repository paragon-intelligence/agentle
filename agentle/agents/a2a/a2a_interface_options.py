from rsb.models.base_model import BaseModel


class A2AInterfaceOptions(BaseModel):
    max_concurrent_tasks: int
    task_timeout: int
    task_poll_interval: int
    keep_completed_tasks: int
