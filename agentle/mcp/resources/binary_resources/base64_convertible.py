"""
Base64Convertible Abstract Base Class Module for MCP.

This module defines the Base64Convertible abstract base class which provides an
interface for objects that can be converted to base64-encoded representation in the
Model Control Protocol (MCP) system.
"""

import abc


class Base64Convertible(abc.ABC):
    """
    Abstract base class for objects that can be converted to base64 encoding.

    Base64Convertible defines a common interface for any object that can provide
    a base64-encoded representation of itself. Classes implementing this interface
    can be used as sources for binary resources in the MCP system.

    Implementations must provide a get_base64 method that returns the base64-encoded
    representation of the object.
    """

    @abc.abstractmethod
    def get_base64(self) -> str:
        """
        Returns the base64-encoded representation of the object.

        Returns:
            str: The base64-encoded representation of this object
        """
        ...
