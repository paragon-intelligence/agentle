"""Tests for the DocumentKnowledge class."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

from agentle.agents.knowledge.document_knowledge import DocumentKnowledge


class TestDocumentKnowledge:
    """Tests for the DocumentKnowledge class."""

    def test_document_knowledge_initialization(self):
        """Tests that a DocumentKnowledge object can be initialized with a file path."""
        # Initialize with a string path
        knowledge1 = DocumentKnowledge(file_path="test_document.pdf")
        assert knowledge1.file_path == "test_document.pdf"

        # Initialize with a Path object
        path_obj = Path("another_document.txt")
        knowledge2 = DocumentKnowledge(file_path=path_obj)
        assert knowledge2.file_path == path_obj

    def test_document_knowledge_str_representation(self):
        """Tests the string representation of DocumentKnowledge."""
        knowledge = DocumentKnowledge(file_path="test_document.pdf")

        # str() should return the file path
        assert str(knowledge) == "test_document.pdf"

    @patch("agentle.parsing.document_parser.DocumentParser.parse")
    def test_document_knowledge_parse(self, mock_parse):
        """Tests that a DocumentKnowledge object can be parsed."""
        mock_parse.return_value = "Parsed document content"
        mock_parser = MagicMock()
        mock_parser.parse.return_value = "Parsed document content"

        knowledge = DocumentKnowledge(file_path="test_document.pdf")

        # Call parse directly for testing
        result = knowledge.parse(parser=mock_parser)

        # Verify that the parser was called with the file path
        mock_parser.parse.assert_called_once_with("test_document.pdf")

        # Check the result
        assert result == "Parsed document content"

    @pytest.mark.asyncio
    async def test_document_knowledge_parse_async(self):
        """Tests that a DocumentKnowledge object can be parsed asynchronously."""
        mock_parser = MagicMock()
        mock_parser.parse_async = MagicMock()
        mock_parser.parse_async.return_value = "Parsed document content async"

        knowledge = DocumentKnowledge(file_path="test_document.pdf")

        # Call parse_async directly for testing
        result = await knowledge.parse_async(parser=mock_parser)

        # Verify that the parser was called with the file path
        mock_parser.parse_async.assert_called_once_with("test_document.pdf")

        # Check the result
        assert result == "Parsed document content async"
