from typing import Literal

from agentle.mcp.models.annotations import Annotations
from agentle.mcp.models.blob_resource_contents import BlobResourceContents
from agentle.mcp.models.text_resource_contents import TextResourceContents
from rsb.models.base_model import BaseModel
from rsb.models.config_dict import ConfigDict
from rsb.models.field import Field


class EmbeddedResource(BaseModel):
    """
    The contents of a resource, embedded into a prompt or tool call result.

    It is up to the client how best to render embedded resources for the benefit
    of the LLM and/or the user.
    """

    type: Literal["resource"] = Field(default="resource")
    resource: TextResourceContents | BlobResourceContents = Field(discriminator="type")
    annotations: Annotations | None = None
    model_config = ConfigDict(extra="allow")
