from pathlib import Path
from typing import override

from agentle.parsing.document_parser import DocumentParser
from agentle.parsing.parsed_document import ParsedDocument
from agentle.parsing.parses import parses
from agentle.parsing.section_content import SectionContent


@parses("txt", "alg")
class TxtFileParser(DocumentParser):
    @override
    async def parse_async(self, document_path: str) -> ParsedDocument:
        path = Path(document_path)
        text_content = path.read_text(encoding="utf-8", errors="replace")

        page_content = SectionContent(
            number=1,
            text=text_content,
            md=text_content,
        )

        return ParsedDocument(
            name=path.name,
            sections=[page_content],
        )
