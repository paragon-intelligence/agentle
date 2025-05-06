"""
XML File Parser Module

This module provides functionality for parsing XML files into structured document
representations, converting XML structures into readable markdown format.
"""

import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Literal, override
from xml.etree.ElementTree import Element

from rsb.models.config_dict import ConfigDict
from rsb.models.field import Field

from agentle.parsing.document_parser import DocumentParser
from agentle.parsing.parsed_document import ParsedDocument
from agentle.parsing.section_content import SectionContent

logger = logging.getLogger(__name__)


class XMLFileParser(DocumentParser):
    """
    Parser for processing XML files into structured document representations.

    This parser reads XML files and converts them into a ParsedDocument representation,
    transforming the XML structure into a nested markdown format for better readability.
    The parser attempts to preserve the hierarchical structure of the XML by using
    indented markdown lists, making complex XML documents easier to navigate and understand.

    **Attributes:**

    *   `type` (Literal["xml"]):
        Constant that identifies this as an XML parser. Always set to "xml".

    **Usage Examples:**

    Basic parsing of an XML file:
    ```python
    from agentle.parsing.parsers.xml import XMLFileParser

    # Create a parser
    parser = XMLFileParser()

    # Parse an XML file
    parsed_doc = parser.parse("config.xml")

    # Access the structured content
    print(parsed_doc.sections[0].md)  # Outputs formatted markdown representation
    ```

    Using the parser through the facade:
    ```python
    from agentle.parsing.parse import parse

    # Parse an XML file
    parsed_doc = parse("data.xml")

    # Get original XML content
    raw_xml = parsed_doc.sections[0].text

    # Get markdown representation
    markdown_format = parsed_doc.sections[0].md

    print(f"First 100 chars of markdown representation:\n{markdown_format[:100]}...")
    ```
    """

    type: Literal["xml"] = Field(default="xml")

    @override
    async def parse_async(self, document_path: str) -> ParsedDocument:
        """
        Asynchronously parse an XML file into a structured representation.

        This method reads an XML file, attempts to parse its structure, and converts it
        into a ParsedDocument with a single section containing both the raw XML text
        and a markdown representation of the XML structure.

        Args:
            document_path (str): Path to the XML file to be parsed

        Returns:
            ParsedDocument: A structured representation of the XML file with:
                - text: The raw XML text content
                - md: A markdown representation of the XML structure

        Example:
            ```python
            import asyncio
            from agentle.parsing.parsers.xml import XMLFileParser

            async def process_xml_file():
                parser = XMLFileParser()
                result = await parser.parse_async("settings.xml")

                # Print the formatted markdown representation
                print(result.sections[0].md)

            asyncio.run(process_xml_file())
            ```
        """
        file = Path(document_path)
        raw_xml = file.read_bytes().decode("utf-8", errors="replace")
        md_content = self.xml_to_md(raw_xml)

        section_content = SectionContent(
            number=1,
            text=raw_xml,
            md=md_content,
            images=[],
            items=[],
        )

        return ParsedDocument(
            name=file.name,
            sections=[section_content],
        )

    def xml_to_md(self, xml_str: str) -> str:
        """
        Converts XML content into a nested Markdown list structure.

        This method parses XML content and transforms it into a markdown format
        that preserves the hierarchical structure by using nested lists with
        appropriate indentation. The resulting markdown is more readable and
        navigable than raw XML.

        Args:
            xml_str (str): XML content as a string

        Returns:
            str: Markdown representation of the XML content, or raw XML in a code
                block if parsing fails

        Example:
            ```python
            from agentle.parsing.parsers.xml import XMLFileParser

            parser = XMLFileParser()
            xml_content = '<root><item>Value</item><nested><child>Data</child></nested></root>'
            markdown = parser.xml_to_md(xml_content)
            print(markdown)
            # Output:
            # - **root**
            #   - **item**
            #     - *Text*: Value
            #   - **nested**
            #     - **child**
            #       - *Text*: Data
            ```
        """
        try:
            root: Element = ET.fromstring(xml_str)
            return self._convert_element_to_md(root, level=0)
        except ET.ParseError as e:
            logger.exception("Error parsing XML: %s", e)
            return "```xml\n" + xml_str + "\n```"  # Fallback to raw XML in code block

    def _convert_element_to_md(self, element: Element, level: int) -> str:
        """
        Recursively converts an XML element and its children to Markdown.

        This helper method handles the conversion of a single XML element to markdown,
        then recursively processes all child elements, maintaining appropriate indentation
        for each level of nesting.

        Args:
            element (Element): The XML element to convert
            level (int): Current nesting level for indentation

        Returns:
            str: Markdown representation of the element and its children

        Note:
            This method is intended for internal use by the xml_to_md method.
        """
        indent = "  " * level
        lines: list[str] = []

        # Element tag as bold item
        lines.append(f"{indent}- **{element.tag}**")

        # Attributes as sub-items
        if element.attrib:
            lines.append(f"{indent}  - *Attributes*:")
            for key, value in element.attrib.items():
                lines.append(f"{indent}    - `{key}`: `{value}`")

        # Text content
        if element.text and element.text.strip():
            text = element.text.strip().replace("\n", " ")
            lines.append(f"{indent}  - *Text*: {text}")

        # Process child elements recursively
        for child in element:
            lines.append(self._convert_element_to_md(child, level + 1))

        return "\n".join(lines)

    model_config = ConfigDict(frozen=True)
