import logging
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Literal, override
from xml.etree.ElementTree import Element

from rsb.models.config_dict import ConfigDict
from rsb.models.field import Field

from agentle.parsing.document_parser import DocumentParser
from agentle.parsing.parsed_document import ParsedDocument
from agentle.parsing.parses import parses
from agentle.parsing.section_content import SectionContent

logger = logging.getLogger(__name__)

@parses("xml")
class XMLFileParser(DocumentParser):
    type: Literal["xml"] = Field(default="xml")

    @override
    async def parse_async(self, document_path: str) -> ParsedDocument:
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
        """Converts XML content into a nested Markdown list structure."""
        try:
            root: Element = ET.fromstring(xml_str)
            return self._convert_element_to_md(root, level=0)
        except ET.ParseError as e:
            logger.exception("Error parsing XML: %s", e)
            return "```xml\n" + xml_str + "\n```"  # Fallback to raw XML in code block

    def _convert_element_to_md(self, element: Element, level: int) -> str:
        """Recursively converts an XML element and its children to Markdown.

        Args:
            element: The XML element to convert
            level: Current nesting level for indentation
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
