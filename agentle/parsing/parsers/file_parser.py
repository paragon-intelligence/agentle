import inspect
from pathlib import Path
from typing import Any, Literal, MutableMapping, cast, override
from urllib.parse import urlparse

from rsb.functions.create_instance_dynamically import create_instance_dynamically
from rsb.models.field import Field

from agentle.agents.agent import Agent
from agentle.generations.models.structured_outputs_store.audio_description import (
    AudioDescription,
)
from agentle.generations.models.structured_outputs_store.visual_media_description import (
    VisualMediaDescription,
)
from agentle.parsing.document_parser import DocumentParser
from agentle.parsing.factories.audio_description_agent_default_factory import (
    audio_description_agent_default_factory,
)
from agentle.parsing.factories.visual_description_agent_default_factory import (
    visual_description_agent_default_factory,
)
from agentle.parsing.parsed_document import ParsedDocument


class FileParser(DocumentParser):
    """
    A facade parser that automatically selects the appropriate parser based on file extension.

    The FileParser class acts as a smart entry point to the parsing system, dynamically
    selecting and configuring the appropriate parser based on a document's file extension.
    This eliminates the need for users to know which specific parser to use for each file
    type, making the parsing system much easier to work with.

    FileParser delegates to the specific parser registered for a given file extension,
    passing along appropriate configuration options like the parsing strategy and any
    custom agents for visual or audio content analysis.

    **Attributes:**

    *   `strategy` (Literal["low", "high"]):
        The parsing strategy to use. Defaults to "high".
        - "high": More thorough parsing with intensive operations like OCR and content analysis
        - "low": More efficient parsing that skips some intensive operations

        **Example:**
        ```python
        parser = FileParser(strategy="low")  # Use faster, less intensive parsing
        ```

    *   `visual_description_agent` (Agent[VisualMediaDescription]):
        An optional custom agent for visual media description. If provided, this agent
        will be used instead of the default for analyzing images and visual content.

        **Example:**
        ```python
        from agentle.agents.agent import Agent
        from agentle.generations.models.structured_outputs_store.visual_media_description import VisualMediaDescription

        custom_visual_agent = Agent(
            model="gemini-2.0-pro-vision",
            instructions="Describe technical diagrams with precision",
            response_schema=VisualMediaDescription
        )

        parser = FileParser(visual_description_agent=custom_visual_agent)
        ```

    *   `audio_description_agent` (Agent[AudioDescription]):
        An optional custom agent for audio description. If provided, this agent
        will be used instead of the default for analyzing audio content.

        **Example:**
        ```python
        from agentle.agents.agent import Agent
        from agentle.generations.models.structured_outputs_store.audio_description import AudioDescription

        custom_audio_agent = Agent(
            model="gemini-2.0-flash",
            instructions="Transcribe technical terminology with high accuracy",
            response_schema=AudioDescription
        )

        parser = FileParser(audio_description_agent=custom_audio_agent)
        ```

    **Usage Examples:**

    Basic usage with default settings:
    ```python
    from agentle.parsing.parsers.file_parser import FileParser

    # Create a parser with default settings
    parser = FileParser()

    # Parse different file types with the same parser
    pdf_doc = parser.parse("document.pdf")
    image = parser.parse("diagram.png")
    spreadsheet = parser.parse("data.xlsx")
    ```

    Using different strategies for different files:
    ```python
    # Create parsers with different strategies
    high_detail_parser = FileParser(strategy="high")
    fast_parser = FileParser(strategy="low")

    # Use high detail for important documents
    contract = high_detail_parser.parse("contract.docx")

    # Use fast parsing for initial screening
    screening_results = fast_parser.parse("batch_of_images.zip")
    ```

    Using custom agents:
    ```python
    # Create custom agents for specialized parsing
    technical_visual_agent = Agent(
        model="gemini-2.0-pro-vision",
        instructions="Focus on technical details in diagrams and charts",
        response_schema=VisualMediaDescription
    )

    legal_audio_agent = Agent(
        model="gemini-2.0-flash",
        instructions="Transcribe legal terminology with high accuracy",
        response_schema=AudioDescription
    )

    # Create a parser with custom agents
    specialized_parser = FileParser(
        visual_description_agent=technical_visual_agent,
        audio_description_agent=legal_audio_agent
    )

    # Parse files with specialized agents
    technical_diagram = specialized_parser.parse("circuit_diagram.png")
    legal_recording = specialized_parser.parse("deposition.mp3")
    ```
    """

    type: Literal["file"] = "file"
    strategy: Literal["low", "high"] = Field(default="high")
    visual_description_agent: Agent[VisualMediaDescription] | None = Field(
        default=None,
    )
    """
    The agent to use for generating the visual description of the document.
    Useful when you want to customize the prompt for the visual description.
    """

    audio_description_agent: Agent[AudioDescription] | None = Field(
        default=None,
    )
    """
    The agent to use for generating the audio description of the document.
    Useful when you want to customize the prompt for the audio description.
    """

    parse_timeout: float = Field(default=30)
    """The timeout for the parse operation in seconds."""

    @override
    async def parse_async(self, document_path: str) -> ParsedDocument:
        """
        Asynchronously parse a document using the appropriate parser for its file type.

        This method examines the file extension of the provided document path, selects
        the appropriate parser for that file type, and delegates the parsing process to
        that specific parser instance. It automatically passes along configuration options
        like the parsing strategy and any custom agents to the selected parser.

        Args:
            document_path (str): Path to the document file to be parsed

        Returns:
            ParsedDocument: A structured representation of the parsed document

        Raises:
            ValueError: If the file extension is not supported by any registered parser

        Example:
            ```python
            import asyncio
            from agentle.parsing.parsers.file_parser import FileParser

            async def process_documents():
                parser = FileParser(strategy="high")

                # Parse different document types
                pdf_result = await parser.parse_async("report.pdf")
                image_result = await parser.parse_async("chart.png")
                spreadsheet_result = await parser.parse_async("data.xlsx")

                # Process the parsed results
                for doc in [pdf_result, image_result, spreadsheet_result]:
                    print(f"Document: {doc.name}")
                    print(f"Section count: {len(doc.sections)}")

            asyncio.run(process_documents())
            ```
        """
        from agentle.parsing.parsers.link import LinkParser
        from agentle.parsing.parses import parser_registry

        path = Path(document_path)
        parser_cls: type[DocumentParser] | None = parser_registry.get(
            path.suffix.lstrip(".")
        )

        visual_description_agent = (
            self.visual_description_agent or visual_description_agent_default_factory()
        )
        audio_description_agent = (
            self.audio_description_agent or audio_description_agent_default_factory()
        )

        if not parser_cls:
            parsed_url = urlparse(document_path)
            is_url = parsed_url.scheme in ["http", "https"]

            if is_url:
                parser_cls = cast(
                    type[DocumentParser],
                    LinkParser,
                )

                return await create_instance_dynamically(  # used because mypy complained about the type of the parser_cls
                    parser_cls,
                    visual_description_agent=visual_description_agent,
                    audio_description_agent=audio_description_agent,
                    parse_timeout=self.parse_timeout,
                ).parse_async(document_path=document_path)
            else:
                raise ValueError(f"Unsupported extension: {path.suffix}")

        # Get the signature of the parser constructor
        parser_signature = inspect.signature(parser_cls.__init__)
        valid_params = parser_signature.parameters.keys()

        # Only include arguments that are accepted by the parser constructor
        potential_args = {
            "strategy": self.strategy,
            "visual_description_agent": self.visual_description_agent,
            "audio_description_agent": self.audio_description_agent,
        }

        kwargs: MutableMapping[str, Any] = {}
        for arg_name, arg_value in potential_args.items():
            if arg_name in valid_params:
                kwargs[arg_name] = arg_value

        return await parser_cls(**kwargs).parse_async(document_path)
