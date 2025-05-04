import os
import tempfile
from collections.abc import MutableSequence
from pathlib import Path
from typing import override

from rsb.models.field import Field

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
from agentle.parsing.parsers.static_image import StaticImageParser
from agentle.parsing.parses import parses
from agentle.parsing.section_content import SectionContent


@parses("dwg")
class DWGFileParser(DocumentParser):
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

    @override
    async def parse_async(self, document_path: str) -> ParsedDocument:
        """
        DWG files are kind of tricky. To parse them, Agentle converts them to PDF first,
        then takes a "screenshot" of each page of the PDF and uses GenAI to describe the images.
        """
        import platform

        if platform.machine() == "arm64":
            raise ValueError(
                "ARM architecture is not supported by aspose-cad. Sorry :("
            )

        import aspose.cad as cad  # type: ignore

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = f"{temp_dir}/{os.path.basename(document_path)}"
            # Save the file to the temp directory
            with open(file_path, "wb") as f:
                f.write(Path(document_path).read_bytes())

            # Load the DWG file
            image = cad.Image.load(file_path)  # type: ignore

            # Specify PDF Options
            pdfOptions = cad.imageoptions.PdfOptions()  # type: ignore

            output_path = f"{temp_dir}/output.pdf"

            # Save as PDF
            image.save(output_path, pdfOptions)  # type: ignore

            # Convert PDF to images and save them to temp directory
            image_paths = self.__pdf_to_image_paths(output_path, temp_dir)

            parser = StaticImageParser(
                visual_description_agent=self.visual_description_agent
            )

            parsed_files = [
                await parser.parse_async(image_path) for image_path in image_paths
            ]
            sections: MutableSequence[SectionContent] = [
                section
                for parsed_file in parsed_files
                for section in parsed_file.sections
            ]

            return ParsedDocument.from_sections(document_path, sections)

    def __pdf_to_image_paths(self, pdf_path: str, temp_dir: str) -> list[str]:
        """Converts each page of a PDF to image files and returns their paths.

        Args:
            pdf_path (str): The path to the PDF file.
            temp_dir (str): The temporary directory to save the images.

        Returns:
            list[str]: A list of file paths to the saved images.
        """
        import pymupdf

        image_paths: list[str] = []
        doc = pymupdf.open(pdf_path)
        base_filename = os.path.basename(pdf_path).split(".")[0]

        try:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)  # type: ignore
                pix = page.get_pixmap()  # type: ignore

                # Create the image file path
                image_path = os.path.join(
                    temp_dir, f"{base_filename}_page_{page_num}.png"
                )

                # Save the image to the temp directory
                pix.save(image_path, "png")  # type: ignore

                image_paths.append(image_path)

        finally:
            doc.close()

        return image_paths
