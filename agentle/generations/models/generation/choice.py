from agentle.generations.models.messages.generated_assistant_message import (
    GeneratedAssistantMessage,
)
from rsb.decorators.value_objects import valueobject
from rsb.models.base_model import BaseModel
from rsb.models.field import Field


@valueobject
class Choice[T](BaseModel):
    index: int = Field(description="The index of the choice.")

    message: GeneratedAssistantMessage[T] = Field(
        description="The message of the Choice.", kw_only=False
    )
