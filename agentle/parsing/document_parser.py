import abc
from rsb.coroutines.run_sync import run_sync
from rsb.models.base_model import BaseModel
from rsb.models.config_dict import ConfigDict


from agentle.parsing.parsed_document import ParsedDocument


class DocumentParser(BaseModel, abc.ABC):
    """
    Abstract base class for all document parsers in the Agentle framework.

    This class defines the interface that all document parsers must implement.
    Document parsers are responsible for taking a document file path, parsing its
    contents, and returning a structured ParsedDocument object representing the
    document's content.

    Different file types will have different parser implementations that extend
    this base class, each specialized for handling specific file formats.

    The DocumentParser class follows a consistent pattern:
    1. It provides both synchronous and asynchronous parsing methods
    2. The synchronous method is a wrapper around the asynchronous one
    3. Subclasses need only implement the async version

    Subclasses of DocumentParser are registered with specific file extensions
    using the `@parses` decorator, allowing the system to automatically select
    the appropriate parser based on file type.

    **Usage Examples:**

    ```python
    # Using a concrete parser implementation directly
    from agentle.parsing.parsers.pdf import PDFFileParser

    pdf_parser = PDFFileParser()
    parsed_document = pdf_parser.parse("path/to/document.pdf")

    # The facade pattern (easier approach)
    from agentle.parsing.parse import parse

    parsed_document = parse("path/to/any_supported_file.ext")
    ```

    **Implementing a Custom Parser:**

    ```python
    from agentle.parsing.document_parser import DocumentParser
    from agentle.parsing.parsed_document import ParsedDocument
    from agentle.parsing.parses import parses
    from agentle.parsing.section_content import SectionContent

    @parses("xyz")  # Register this parser for .xyz files
    class XYZFileParser(DocumentParser):
        async def parse_async(self, document_path: str) -> ParsedDocument:
            # Custom logic to parse .xyz files
            # ...

            return ParsedDocument(
                name=document_path,
                sections=[
                    SectionContent(
                        number=1,
                        text="Parsed content from XYZ file",
                        md="# Parsed content from XYZ file"
                    )
                ]
            )
    ```
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        frozen=True,
    )

    def parse(self, document_path: str) -> ParsedDocument:
        """
        Parse a document synchronously and return a structured representation.

        This method is a synchronous wrapper around the asynchronous parse_async method.
        It provides a convenient interface for parsing documents when you don't need
        asynchronous execution.

        Args:
            document_path (str): Path to the document file to be parsed

        Returns:
            ParsedDocument: A structured representation of the parsed document

        Example:
            ```python
            from agentle.parsing.parsers.pdf import PDFFileParser

            parser = PDFFileParser()
            parsed_doc = parser.parse("document.pdf")

            print(parsed_doc.name)
            for section in parsed_doc.sections:
                print(f"Section {section.number}: {section.text}")
            ```
        """
        return run_sync(self.parse_async, document_path=document_path)

    @abc.abstractmethod
    async def parse_async(self, document_path: str) -> ParsedDocument:
        """
        Parse a document asynchronously and return a structured representation.

        This abstract method must be implemented by all concrete subclasses.
        It should contain the specific logic for parsing a particular document format
        and transforming it into a standardized ParsedDocument structure.

        Args:
            document_path (str): Path to the document file to be parsed

        Returns:
            ParsedDocument: A structured representation of the parsed document

        Raises:
            NotImplementedError: If this method is not overridden by a subclass

        Example:
            ```python
            from agentle.parsing.parsers.pdf import PDFFileParser
            import asyncio

            async def main():
                parser = PDFFileParser()
                parsed_doc = await parser.parse_async("document.pdf")

                print(parsed_doc.name)
                for section in parsed_doc.sections:
                    print(f"Section {section.number}: {section.text}")

            asyncio.run(main())
            ```
        """
        ...
