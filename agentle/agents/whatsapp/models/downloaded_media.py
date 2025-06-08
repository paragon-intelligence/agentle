from rsb.models.base_model import BaseModel
from rsb.models.config_dict import ConfigDict


class DownloadedMedia(BaseModel):
    data: bytes
    mime_type: str
    model_config = ConfigDict(frozen=True)
