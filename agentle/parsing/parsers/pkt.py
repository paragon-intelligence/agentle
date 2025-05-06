"""
Packet Tracer File Parser Module

This module provides functionality for parsing Cisco Packet Tracer files (.pkt, .pka) into
structured representations. It can decompress and extract the XML content from Packet Tracer
files, making the network topology and configuration data accessible.
"""

import tempfile
from pathlib import Path
from typing import Literal, override

from agentle.parsing.document_parser import DocumentParser
from agentle.parsing.parsed_document import ParsedDocument
from agentle.parsing.section_content import SectionContent


class PKTFileParser(DocumentParser):
    """
    Parser for processing Cisco Packet Tracer files (.pkt).

    This parser extracts the underlying XML content from Packet Tracer files, which are
    compressed and encoded network simulation files used by Cisco Packet Tracer.
    The parser decodes the proprietary encoding and decompresses the content to reveal
    the XML structure that defines the network topology, device configurations, and
    simulation parameters.

    **Usage Examples:**

    Basic parsing of a Packet Tracer file:
    ```python
    from agentle.parsing.parsers.pkt import PKTFileParser

    # Create a parser
    parser = PKTFileParser()

    # Parse a Packet Tracer file
    parsed_file = parser.parse("network_topology.pkt")

    # Access the XML content
    xml_content = parsed_file.sections[0].text
    print(f"XML content length: {len(xml_content)}")

    # Check for network components
    if "<router>" in xml_content:
        print("Contains router configurations")
    if "<switch>" in xml_content:
        print("Contains switch configurations")
    ```

    Using the generic parse function:
    ```python
    from agentle.parsing.parse import parse

    # Parse a Packet Tracer file
    result = parse("lab_exercise.pkt")

    # Access the XML content
    xml_content = result.sections[0].text
    print(f"Extracted {len(xml_content)} bytes of XML data")
    ```
    """

    type: Literal["pkt"] = "pkt"

    @override
    async def parse_async(self, document_path: str) -> ParsedDocument:
        """
        Asynchronously parse a Packet Tracer file and extract its XML content.

        This method reads a Packet Tracer file, decodes the proprietary encoding,
        decompresses the content, and extracts the XML data that defines the
        network topology and configuration.

        Args:
            document_path (str): Path to the Packet Tracer file to be parsed

        Returns:
            ParsedDocument: A structured representation containing the extracted
                XML content in a single section

        Example:
            ```python
            import asyncio
            from agentle.parsing.parsers.pkt import PKTFileParser

            async def process_pkt_file():
                parser = PKTFileParser()
                result = await parser.parse_async("network_simulation.pkt")

                # Get the XML content
                xml_data = result.sections[0].text

                # Check for specific network elements
                if "<hostname>" in xml_data:
                    print("Found hostname configurations")

            asyncio.run(process_pkt_file())
            ```

        Note:
            This parser creates a temporary file during processing, which is automatically
            cleaned up after the parsing is complete.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = f"{temp_dir}/{document_path}"
            # file.save_to_file(file_path)
            # TODO: save the file to the temp directory
            with open(file_path, "wb") as f:
                f.write(Path(document_path).read_bytes())

            xml_bytes = self.pkt_to_xml_bytes(file_path)

            # For now, we'll just return the XML content as a single page
            xml_text = xml_bytes.decode("utf-8", errors="replace")

            page_content = SectionContent(
                number=1,
                text=xml_text,
                md=xml_text,
                images=[],
                items=[],
            )

            return ParsedDocument(
                name=document_path,
                sections=[page_content],
            )

    def pkt_to_xml_bytes(self, pkt_file: str) -> bytes:
        """
        Convert a Packet Tracer file (.pkt/.pka) to its XML representation as bytes.

        This method decodes and decompresses a Packet Tracer file to extract its
        underlying XML content. Packet Tracer files use a proprietary format where
        the content is encoded (each byte XORed with a decreasing file length) and
        then compressed using zlib.

        Args:
            pkt_file (str): Path to the input .pkt or .pka file

        Returns:
            bytes: The uncompressed XML content as bytes

        Example:
            ```python
            from agentle.parsing.parsers.pkt import PKTFileParser

            parser = PKTFileParser()
            xml_bytes = parser.pkt_to_xml_bytes("network.pkt")

            # Convert to string
            xml_text = xml_bytes.decode("utf-8")
            print(xml_text[:500])  # Print first 500 chars
            ```

        Note:
            The decoding algorithm reverses the proprietary encoding used by Cisco
            Packet Tracer: it XORs each byte with a decreasing counter and then
            decompresses the result using zlib.
        """
        import zlib

        with open(pkt_file, "rb") as f:
            in_data = bytearray(f.read())

        i_size = len(in_data)
        out = bytearray()

        # Decrypt each byte with decreasing file length
        for byte in in_data:
            out.append(byte ^ (i_size & 0xFF))
            i_size -= 1

        # The first 4 bytes (big-endian) represent the size of the XML when uncompressed
        # (This value is not needed for the actual return, but we parse it for completeness.)
        _uncompressed_size = int.from_bytes(out[:4], byteorder="big")

        # Decompress the data after the first 4 bytes
        xml_data = zlib.decompress(out[4:])

        return xml_data
