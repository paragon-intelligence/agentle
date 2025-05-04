from pathlib import Path
from typing import Literal, Self

from rsb.models.base_model import BaseModel
from rsb.models.field import Field
from rsb.models.model_validator import model_validator


class DocumentKnowledge(BaseModel):
    """
    A document to be used by the agent.
    """

    type: Literal["document"] = Field(default="document")
    path: str = Field(
        description="The path to the document. Can be a url or a local file path."
    )

    @model_validator(mode="after")
    def validate_path(self) -> Self:
        """
        validate if it's a path of file or url. if it's a url, it should be a valid url.
        if it's a path, it should be a valid path to a file.
        """

        if not self.path.startswith("http"):
            if not Path(self.path).exists():
                raise ValueError(f"Document path {self.path} does not exist.")
        return self
