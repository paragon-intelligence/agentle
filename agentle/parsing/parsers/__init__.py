"""
Document Parser Implementations for Various File Types

This package contains parser implementations for different file formats, each extending
the base DocumentParser class. Each parser is designed to extract structured content
from a specific file type and transform it into a standardized ParsedDocument format.

The parsers in this package can handle:

- Text files (TXT, ALG)
- Documents (PDF, DOCX)
- Images (PNG, JPG, TIFF, etc.)
- Audio files (MP3, WAV, etc.)
- Video files (MP4)
- Presentations (PPT, PPTX)
- Spreadsheets (XLS, XLSX)
- Compressed archives (ZIP, RAR)
- Specialized formats (DWG, PKT, XML)
- And more

These parsers are registered via the `@parses` decorator, which maps file extensions
to the appropriate parser class. This allows the system to automatically select the
correct parser based on the file extension.

The most convenient way to use these parsers is through the `parse()` function:

```python
from agentle.parsing.parse import parse

# Parse a document with default settings
parsed_doc = parse("path/to/document.pdf")

# Parse with a specific strategy
parsed_doc = parse("path/to/image.jpg", strategy="high")
```

For more control, you can instantiate and use a specific parser directly:

```python
from agentle.parsing.parsers.pdf import PDFFileParser

# Create a parser instance
pdf_parser = PDFFileParser()

# Parse a PDF document
parsed_doc = pdf_parser.parse("path/to/document.pdf")
```

You can also use the FileParser facade to automatically select the right parser:

```python
from agentle.parsing.parsers.file_parser import FileParser

# Create a facade parser
parser = FileParser()

# Parse any supported file type
parsed_doc = parser.parse("path/to/file.extension")
```
"""
