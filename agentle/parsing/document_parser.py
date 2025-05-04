from rsb.coroutines.run_sync import run_sync
from rsb.models.base_model import BaseModel
from rsb.models.config_dict import ConfigDict


from agentle.parsing.parsed_document import ParsedDocument


class DocumentParser(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    def parse(self, document_path: str) -> ParsedDocument:
        """
        Parse a document and return a ParsedDocument.
        """
        return run_sync(self.parse_async, document_path=document_path)

    async def parse_async(self, document_path: str) -> ParsedDocument:
        """
        Parse a document and return a ParsedDocument.
        """
        raise NotImplementedError("Subclasses must implement this method.")
