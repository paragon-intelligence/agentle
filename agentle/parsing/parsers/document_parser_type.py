from __future__ import annotations

from agentle.parsing.parsers.audio import AudioFileParser
from agentle.parsing.parsers.compressed import CompressedFileParser
from agentle.parsing.parsers.docx import DocxFileParser
from agentle.parsing.parsers.dwg import DWGFileParser
from agentle.parsing.parsers.file_parser import FileParser
from agentle.parsing.parsers.gif import GifFileParser
from agentle.parsing.parsers.html import HTMLParser
from agentle.parsing.parsers.link import LinkParser
from agentle.parsing.parsers.pdf import PDFFileParser
from agentle.parsing.parsers.pkt import PKTFileParser
from agentle.parsing.parsers.pptx import PptxFileParser
from agentle.parsing.parsers.static_image import StaticImageParser
from agentle.parsing.parsers.txt import TxtFileParser
from agentle.parsing.parsers.video import VideoFileParser
from agentle.parsing.parsers.xlsx import XlsxFileParser
from agentle.parsing.parsers.xml import XMLFileParser

type DocumentParserType = (
    AudioFileParser
    | CompressedFileParser
    | DocxFileParser
    | DWGFileParser
    | GifFileParser
    | FileParser
    | HTMLParser
    | LinkParser
    | PDFFileParser
    | PKTFileParser
    | PptxFileParser
    | StaticImageParser
    | TxtFileParser
    | VideoFileParser
    | XlsxFileParser
    | XMLFileParser
)
