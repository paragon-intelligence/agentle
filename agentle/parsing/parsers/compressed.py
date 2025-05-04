from collections.abc import MutableSequence
from pathlib import Path
from typing import Self, cast, override

from rsb.models.field import Field
from rsb.models.model_validator import model_validator

from agentle.agents.agent import Agent
from agentle.generations.models.structured_outputs_store.visual_media_description import (
    VisualMediaDescription,
)
from agentle.generations.providers.base.generation_provider import GenerationProvider
from agentle.generations.providers.google.google_genai_generation_provider import (
    GoogleGenaiGenerationProvider,
)
from agentle.parsing.document_parser import DocumentParser
from agentle.parsing.factories.visual_description_agent_factory import (
    visual_description_agent_factory,
)
from agentle.parsing.parsed_document import ParsedDocument
from agentle.parsing.parsers.file_parser import FileParser
from agentle.parsing.parsers.validate_visual_parsers import validate_visual_parsers
from agentle.parsing.parses import parses


@parses("zip", "rar", "pkz")
class CompressedFileParser(DocumentParser):
    inner_parser: DocumentParser = Field(
        default_factory=FileParser,
    )
    """
    The inner parser to use for parsing the compressed files.
    Defaults to a facade that automatically selects the correct parser based on the file extension.
    """

    visual_description_agent: Agent[VisualMediaDescription] = Field(
        default_factory=visual_description_agent_factory,
    )
    """
    The agent to use for generating the visual description of the document.
    Useful when you want to customize the prompt for the visual description.
    """

    multi_modal_provider: GenerationProvider = Field(
        default_factory=GoogleGenaiGenerationProvider,
    )
    """
    The multi-modal provider to use for generating the visual description of the document.
    Useful when you want us to customize the prompt for the visual description.
    """

    @model_validator(mode="after")
    def validate_visual_parsers(self) -> Self:
        validate_visual_parsers(
            self.visual_description_agent, self.multi_modal_provider
        )
        return self

    @override
    async def parse_async(self, document_path: str) -> ParsedDocument:
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
