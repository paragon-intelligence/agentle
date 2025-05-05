"""
THE PARSING MODULE ENABLES AGENTS TO BE TRUE MULTI-MODAL.

This module provides a comprehensive framework for parsing and processing various file types
into a structured format that can be used by AI agents. It supports:

- Document parsing (PDF, DOCX, TXT, etc.)
- Image processing (PNG, JPG, GIF, etc.)
- Audio file analysis (MP3, WAV, etc.)
- Video content processing (MP4)
- Spreadsheet data extraction (XLS, XLSX)
- Presentation parsing (PPT, PPTX)
- Compressed file handling (ZIP, RAR)
- And more specialized formats

The module uses a registry pattern to map file extensions to appropriate parsers,
making it easy to add support for new file types.

Basic usage:
```python
from agentle.parsing.parse import parse

# Parse a document with default settings
parsed_doc = parse("path/to/document.pdf")

# Access the parsed content
print(parsed_doc.name)
for section in parsed_doc.sections:
    print(f"Section {section.number}: {section.text}")
    for image in section.images:
        print(f"Image: {image.name}, OCR text: {image.ocr_text}")
```
"""
