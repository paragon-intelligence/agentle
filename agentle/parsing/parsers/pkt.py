import tempfile
from pathlib import Path
from typing import override

from agentle.parsing.document_parser import DocumentParser
from agentle.parsing.parsed_document import ParsedDocument
from agentle.parsing.parses import parses
from agentle.parsing.section_content import SectionContent


@parses("pkt")
class PKTFileParser(DocumentParser):
    @override
    async def parse_async(self, document_path: str) -> ParsedDocument:
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

        :param pkt_file: Path to the input .pkt or .pka file.
        :return: The uncompressed XML content as bytes.
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
