"""
Compressed Archive Parser Module

This module provides functionality for parsing compressed archive files (ZIP, RAR, PKZ) into
structured representations. It can extract contents from compressed archives and parse the
contained files using appropriate parsers for each file type.
"""

from collections.abc import MutableSequence
from pathlib import Path
from typing import Literal, cast, override

from rsb.models.field import Field

from agentle.agents.agent import Agent
from agentle.generations.models.structured_outputs_store.visual_media_description import (
    VisualMediaDescription,
)
from agentle.parsing.document_parser import DocumentParser
from agentle.parsing.factories.visual_description_agent_default_factory import (
    visual_description_agent_default_factory,
)
from agentle.parsing.parsed_document import ParsedDocument
from agentle.parsing.parsers.file_parser import FileParser


class CompressedFileParser(DocumentParser):
    """
    Parser for processing compressed archive files (ZIP, RAR, PKZ).

    This parser extracts files from compressed archives and processes each file using
    the appropriate parser for its file type. It acts as a container parser that delegates
    the actual parsing work to specialized parsers for each contained file type.
    The results from all contained files are then combined into a single ParsedDocument.

    **Attributes:**

    *   `inner_parser` (DocumentParser):
        The parser to use for parsing files extracted from the archive. Defaults to
        FileParser, which automatically selects the appropriate parser based on file extension.

        **Example:**
        ```python
        from agentle.parsing.parsers.file_parser import FileParser

        # Create a specialized inner parser with custom settings
        inner_parser = FileParser(strategy="low")

        # Use it in the compressed file parser
        parser = CompressedFileParser(inner_parser=inner_parser)
        ```

    *   `visual_description_agent` (Agent[VisualMediaDescription]):
        The agent used to analyze and describe visual content. This agent is passed
        to the inner parsers when processing image files and other visual content.
        Defaults to the agent created by `visual_description_agent_default_factory()`.

        Note: You cannot use both visual_description_agent and multi_modal_provider
        at the same time.

        **Example:**
        ```python
        from agentle.agents.agent import Agent
        from agentle.generations.models.structured_outputs_store.visual_media_description import VisualMediaDescription

        custom_agent = Agent(
            model="gemini-2.0-pro-vision",
            instructions="Focus on technical content in diagrams and charts",
            response_schema=VisualMediaDescription
        )

        parser = CompressedFileParser(visual_description_agent=custom_agent)
        ```

    *   `multi_modal_provider` (GenerationProvider):
        An alternative to using a visual_description_agent. This is a generation
        provider capable of handling multi-modal content (text and images).
        Defaults to GoogleGenaiGenerationProvider().

        Note: You cannot use both visual_description_agent and multi_modal_provider
        at the same time.

    **Usage Examples:**

    Basic parsing of a compressed archive:
    ```python
    from agentle.parsing.parsers.compressed import CompressedFileParser

    # Create a parser with default settings
    parser = CompressedFileParser()

    # Parse a ZIP file
    parsed_archive = parser.parse("documents.zip")

    # Access the combined content from all files
    print(f"Archive contains {len(parsed_archive.sections)} sections")

    # Iterate through all sections from all files
    for i, section in enumerate(parsed_archive.sections):
        print(f"Section {i+1}: {section.text[:100]}...")
    ```

    Using the generic parse function:
    ```python
    from agentle.parsing.parse import parse

    # Parse different archive types
    zip_result = parse("files.zip")
    rar_result = parse("documents.rar")

    # Process the results
    for result in [zip_result, rar_result]:
        print(f"Archive: {result.name}")
        print(f"Contains {len(result.sections)} sections")
    ```
    """

    type: Literal["compressed"] = "compressed"

    inner_parser: DocumentParser = Field(
        default_factory=FileParser,
    )
    """
    The inner parser to use for parsing the compressed files.
    Defaults to a facade that automatically selects the correct parser based on the file extension.
    """

    visual_description_agent: Agent[VisualMediaDescription] = Field(
        default_factory=visual_description_agent_default_factory,
    )
    """
    The agent to use for generating the visual description of the document.
    Useful when you want to customize the prompt for the visual description.
    """

    @override
    async def parse_async(self, document_path: str) -> ParsedDocument:
        """
        Asynchronously parse a compressed archive file and process its contents.

        This method extracts files from a compressed archive (ZIP, RAR, or PKZ),
        processes each file using the appropriate parser for its file type, and
        combines the results into a single ParsedDocument.

        Args:
            document_path (str): Path to the compressed archive file to be parsed

        Returns:
            ParsedDocument: A structured representation combining the parsed content
                from all files in the archive

        Raises:
            ValueError: If the file extension is not supported (not ZIP, RAR, or PKZ)

        Example:
            ```python
            import asyncio
            from agentle.parsing.parsers.compressed import CompressedFileParser

            async def process_archive():
                parser = CompressedFileParser()
                result = await parser.parse_async("documents.zip")

                # Print information about the contents
                print(f"Archive contains content from multiple files")
                print(f"Total sections: {len(result.sections)}")

                # Access the combined content
                for i, section in enumerate(result.sections):
                    print(f"Section {i+1} content type: {type(section).__name__}")
                    if section.images:
                        print(f"  Contains {len(section.images)} images")

            asyncio.run(process_archive())
            ```

        Note:
            This method creates a temporary copy of the archive file for processing.
            The temporary files are automatically cleaned up after processing.
        """
        import tempfile
        import zipfile

        import rarfile

        path = Path(document_path)
        file_contents = path.read_bytes()

        # We'll accumulate ParsedFile objects from each extracted child file
        parsed_files: MutableSequence[ParsedDocument] = []

        # Write the compressed file to a temporary location
        with tempfile.NamedTemporaryFile(delete=True) as tmp:
            tmp.write(file_contents)
            tmp.flush()

            # Decide how to open the archive based on extension
            match path.suffix:
                case "zip" | "pkz":
                    # Treat PKZ exactly like ZIP for demo purposes
                    with zipfile.ZipFile(tmp.name, "r") as zip_ref:
                        # Iterate over files inside the archive
                        for info in zip_ref.infolist():
                            # Directories have filename ending with "/"
                            if info.is_dir():
                                continue

                            # Read raw bytes of the child file
                            child_name = info.filename
                            # Parse using our FileParser façade
                            # (re-using the same strategy/visual_description_agent)
                            parser = FileParser(
                                visual_description_agent=self.visual_description_agent,
                            )
                            child_parsed = await parser.parse_async(child_name)
                            parsed_files.append(child_parsed)

                case "rar":
                    with rarfile.RarFile(tmp.name, "r") as rar_ref:
                        for info in rar_ref.infolist():
                            """Type of "isdir" is unknownPylancereportUnknownMemberType"""
                            if info.isdir():  # type: ignore
                                continue

                            child_name: str = cast(str, info.filename)  # type: ignore

                            parser = FileParser(
                                visual_description_agent=self.visual_description_agent,
                            )
                            child_parsed = await parser.parse_async(child_name)
                            parsed_files.append(child_parsed)

                case _:
                    # Fallback if something else accidentally calls this parser
                    raise ValueError(
                        f"CompressedFileParser does not handle extension: {path.suffix}"
                    )

        # Merge all the parsed files into a single ParsedFile
        return ParsedDocument.from_parsed_files(parsed_files)
