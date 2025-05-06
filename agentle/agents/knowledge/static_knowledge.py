from pathlib import Path
from urllib.parse import urlparse
from rsb.models.base_model import BaseModel
from rsb.models.field import Field
from typing import Literal

NO_CACHE = None

type NoCache = None


class StaticKnowledge(BaseModel):
    """
    Static knowledge is a collection of knowledge that is provided to the agent at the time of creation.
    """

    content: str = Field(
        description="""The content of the knowledge.
        can be a url, a local file path, or a string of text."""
    )

    cache: int | NoCache | Literal["infinite"] = Field(
        default=NO_CACHE,
        description="The cache time of the knowledge. If None, the knowledge is not cached. If 'infinite', the knowledge is cached indefinitely.",
    )

    parse_timeout: float = Field(default=30)
    """The timeout for the parse operation in seconds."""

    def is_url(self) -> bool:
        parsed_url = urlparse(self.content)
        return parsed_url.scheme in ["http", "https"]

    def is_file_path(self) -> bool:
        return Path(self.content).exists()

    def is_raw_text(self) -> bool:
        return not self.is_url() and not self.is_file_path()

    def __str__(self) -> str:
        return self.content
