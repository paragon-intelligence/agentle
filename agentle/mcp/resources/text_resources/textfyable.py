"""
Textfyable Abstract Base Class Module for MCP.

This module defines the Textfyable abstract base class which provides an
interface for objects that can be converted to text representation in the
Model Control Protocol (MCP) system.
"""

import abc


class Textfyable(abc.ABC):
    """
    Abstract base class for objects that can be converted to text.

    Textfyable defines a common interface for any object that can provide
    a textual representation of itself. Classes implementing this interface
    can be used as sources for text resources in the MCP system.

    Implementations must provide a get_text method that returns the text
    representation of the object.
    """

    @abc.abstractmethod
    def get_text(self) -> str:
        """
        Returns the text representation of the object.

        Returns:
            str: The textual representation of this object
        """
        ...
