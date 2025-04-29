from datetime import datetime

from rsb.models.base_model import BaseModel

from agentle.agents.a2a.messages.message import Message
from agentle.agents.a2a.tasks.task_state import TaskState


class TaskStatus(BaseModel):
    state: TaskState
    """
    additional status updates for client
    """

    message: Message
    """
    additional status updates for client
    """

    timestamp: datetime
    """
    ISO datetime value
    """
