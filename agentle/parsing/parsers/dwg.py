import io
import tempfile
from collections.abc import Sequence
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
from agentle.parsing.migrate import StaticImageFileParser
from agentle.parsing.parsed_document import ParsedDocument
from agentle.parsing.parses import parses


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
            file_path = f"{temp_dir}/{document_path}"
            # file.save_to_file(file_path)
            # TODO: save the file to the temp directory
            with open(file_path, "wb") as f:
                f.write(Path(document_path).read_bytes())

            # Load the DWG file
            image = cad.Image.load(file_path)  # type: ignore

            # Specify PDF Options
            pdfOptions = cad.imageoptions.PdfOptions()  # type: ignore

            output_path = f"{temp_dir}/output.pdf"

            # Save as PDF
            image.save(output_path, pdfOptions)  # type: ignore

            image_bytes_list = self.__pdf_to_images(output_path)

            # replace with file paths
            raw_files = [
                RawFile.from_bytes(
                    data=img, name=f"{file.name}_{i}.png", extension="png"
                )
                for i, img in enumerate(image_bytes_list)
            ]

            parser = StaticImageFileParser(
                visual_description_agent=self.visual_description_agent,
            )

            parsed_files = [await parser.parse_async(f) for f in raw_files]
            sections = [
                section
                for parsed_file in parsed_files
                for section in parsed_file.sections
            ]

            return ParsedDocument.from_sections(document_path, sections)

    def __pdf_to_images(self, pdf_path: str) -> Sequence[bytes]:
        """Converts each page of a PDF to image bytes.

        Args:
            pdf_path (str): The path to the PDF file.

        Returns:
            Sequence[bytes]: A list of bytes objects, each containing a PNG image of a PDF page.
        """
        import pymupdf

        image_bytes_list: list[bytes] = []
        doc = pymupdf.open(pdf_path)

        try:
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)  # type: ignore
                pix = page.get_pixmap()  # type: ignore

                # Create a bytes buffer and save the image into it
                buffer = io.BytesIO()
                pix.save(buffer, "png")  # type: ignore
                image_bytes = buffer.getvalue()

                image_bytes_list.append(image_bytes)

        finally:
            doc.close()

        return image_bytes_list
