"""
AutoCAD DWG File Parser Module

This module provides functionality for parsing AutoCAD DWG files into structured
representations. It converts DWG files to PDF and then analyzes the visual content
using AI to generate descriptions of the technical drawings.
"""

import os
import tempfile
from collections.abc import MutableSequence
from pathlib import Path
from typing import Literal, override

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
from agentle.parsing.parsers.static_image import StaticImageParser
from agentle.parsing.section_content import SectionContent


class DWGFileParser(DocumentParser):
    """
    Parser for processing AutoCAD DWG files.

    This parser extracts content from DWG files, which are binary files used by AutoCAD
    for storing 2D and 3D design data. Since DWG files cannot be directly interpreted
    as text or images, the parser converts them to PDF first using Aspose.CAD, then
    renders each page of the PDF as an image. These images are then analyzed using
    a visual description agent to generate textual descriptions of the technical drawings.

    **Attributes:**

    *   `visual_description_agent` (Agent[VisualMediaDescription]):
        The agent used to analyze and describe the visual content of the CAD drawings.
        This agent is responsible for generating descriptions that capture the technical
        details of the drawings.
        Defaults to the agent created by `visual_description_agent_default_factory()`.

        **Example:**
        ```python
        from agentle.agents.agent import Agent
        from agentle.generations.models.structured_outputs_store.visual_media_description import VisualMediaDescription

        custom_agent = Agent(
            model="gemini-2.0-pro-vision",
            instructions="Focus on technical details of engineering drawings and architectural plans",
            response_schema=VisualMediaDescription
        )

        parser = DWGFileParser(visual_description_agent=custom_agent)
        ```

    **Usage Examples:**

    Basic parsing of a DWG file:
    ```python
    from agentle.parsing.parsers.dwg import DWGFileParser

    # Create a parser with default settings
    parser = DWGFileParser()

    # Parse a DWG file
    parsed_drawing = parser.parse("architectural_plan.dwg")

    # Access the parsed content
    for i, section in enumerate(parsed_drawing.sections):
        print(f"Drawing page {i+1}:")
        print(section.text)  # AI-generated description of the drawing
    ```

    Using the generic parse function:
    ```python
    from agentle.parsing.parse import parse

    # Parse a DWG file
    result = parse("mechanical_drawing.dwg")

    # Access the descriptions
    print(f"Drawing contains {len(result.sections)} pages/views")
    for section in result.sections:
        print(section.text)
    ```

    **Requirements:**

    This parser has specific requirements:
    1. The Aspose.CAD library must be installed
    2. ARM architecture (e.g., Apple Silicon) is not supported by Aspose.CAD
    3. PyMuPDF is required for PDF rendering
    """

    type: Literal["dwg"] = "dwg"

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
        Asynchronously parse a DWG file and generate a structured representation.

        This method performs a series of conversions to extract meaningful content from DWG files:
        1. It loads the DWG file using Aspose.CAD
        2. Converts it to PDF format
        3. Renders each page of the PDF as an image
        4. Analyzes each image using a visual description agent to generate descriptions

        Args:
            document_path (str): Path to the DWG file to be parsed

        Returns:
            ParsedDocument: A structured representation where:
                - Each page/view from the drawing is a separate section
                - Each section contains an AI-generated description of the visual content

        Raises:
            ValueError: If running on ARM architecture (not supported by Aspose.CAD)
            RuntimeError: If dependencies are missing or conversion fails

        Example:
            ```python
            import asyncio
            from agentle.parsing.parsers.dwg import DWGFileParser

            async def process_cad_drawing():
                parser = DWGFileParser()
                result = await parser.parse_async("floor_plan.dwg")

                # Print information about the drawing
                print(f"Drawing contains {len(result.sections)} views/pages")

                # Access the descriptions
                for i, section in enumerate(result.sections):
                    print(f"Page {i+1} description:")
                    print(section.text)

            asyncio.run(process_cad_drawing())
            ```

        Note:
            This method is not compatible with ARM architecture (e.g., Apple Silicon)
            due to limitations in the Aspose.CAD library.
        """
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
        """
        Converts each page of a PDF to image files and returns their paths.

        This helper method uses PyMuPDF to render each page of a PDF as a PNG image
        file. It creates one image file for each page of the PDF.

        Args:
            pdf_path (str): The path to the PDF file
            temp_dir (str): The temporary directory to save the images

        Returns:
            list[str]: A list of file paths to the saved images, one per PDF page

        Example:
            ```python
            with tempfile.TemporaryDirectory() as temp_dir:
                # First convert DWG to PDF
                pdf_path = convert_dwg_to_pdf(dwg_file, temp_dir)

                # Then convert PDF pages to images
                image_paths = self.__pdf_to_image_paths(pdf_path, temp_dir)

                # Process each image
                for image_path in image_paths:
                    # Analyze image...
                    pass
            ```

        Note:
            This method requires PyMuPDF to be installed. It is used internally by
            the parse_async method and not typically called directly.
        """
        import pymupdf

        image_paths: list[str] = []
        doc = pymupdf.open(pdf_path)  # type: ignore
        base_filename = os.path.basename(pdf_path).split(".")[0]

        try:
            for page_num in range(len(doc)):  # type: ignore
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
