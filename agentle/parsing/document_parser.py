from rsb.coroutines.run_sync import run_sync
from rsb.models.base_model import BaseModel

from agentle.parsing.parsed_document import ParsedDocument


class DocumentParser(BaseModel):
    """
    A document parser to be used by the agent. This will be used to parse the static
    knowledge.
    """

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
