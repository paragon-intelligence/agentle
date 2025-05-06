"""
Text File Parser Module

This module provides functionality for parsing plain text files (.txt, .alg)
into structured document representations.
"""

from pathlib import Path
from typing import Literal, override

from agentle.parsing.document_parser import DocumentParser
from agentle.parsing.parsed_document import ParsedDocument
from agentle.parsing.section_content import SectionContent


class TxtFileParser(DocumentParser):
    """
    Parser for processing plain text files (.txt, .alg).

    This parser provides a simple implementation for reading text files and converting
    them into a structured ParsedDocument representation. The parser reads the entire
    file content and places it into a single section. It handles both .txt files and
    .alg (algorithm) files.

    This parser is one of the simplest implementations in the framework, as it doesn't
    require any special processing like OCR, media analysis, or structural parsing.

    **Usage Examples:**

    Basic parsing of a text file:
    ```python
    from agentle.parsing.parsers.txt import TxtFileParser

    # Create a parser
    parser = TxtFileParser()

    # Parse a text file
    parsed_doc = parser.parse("notes.txt")

    # Access the text content
    print(parsed_doc.sections[0].text)
    ```

    Using the parser through the facade:
    ```python
    from agentle.parsing.parse import parse

    # Parse a text file using the facade
    parsed_doc = parse("algorithm.alg")

    # Access the content
    content = parsed_doc.sections[0].text
    print(f"Algorithm content:\n{content}")
    ```
    """

    type: Literal["txt"] = "txt"

    @override
    async def parse_async(self, document_path: str) -> ParsedDocument:
        """
        Asynchronously parse a text file into a structured representation.

        This method reads the content of a text file and converts it into a ParsedDocument
        with a single section containing the file's text.

        Args:
            document_path (str): Path to the text file to be parsed

        Returns:
            ParsedDocument: A structured representation of the text file with a
                single section containing the entire file content

        Example:
            ```python
            import asyncio
            from agentle.parsing.parsers.txt import TxtFileParser

            async def process_text_file():
                parser = TxtFileParser()
                result = await parser.parse_async("instructions.txt")
                print(f"File name: {result.name}")
                print(f"Content: {result.sections[0].text}")

            asyncio.run(process_text_file())
            ```

        Note:
            This parser handles UTF-8 encoded text files and uses error replacement
            for any characters that cannot be decoded properly.
        """
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
