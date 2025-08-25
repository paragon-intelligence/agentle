from collections.abc import Sequence

from rsb.models.base_model import BaseModel
from rsb.models.config_dict import ConfigDict
from rsb.models.field import Field

from agentle.agents.whatsapp.structured_outputs.message_output_type import (
    MessageOutputType,
)


class AgentOutput(BaseModel):
    output: Sequence[MessageOutputType] = Field(discriminator="type")

    model_config = ConfigDict(frozen=True)
