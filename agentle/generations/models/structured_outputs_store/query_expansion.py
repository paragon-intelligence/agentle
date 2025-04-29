"""
Module for query expansion models.

This module provides the QueryExpansion class which represents an expanded version
of a search query that may provide additional context or specificity.
"""

from rsb.decorators.value_objects import valueobject
from rsb.models.base_model import BaseModel
from rsb.models.field import Field


@valueobject
class QueryExpansion(BaseModel):
    """
    Represents an expanded version of a search query.

    This class can contain an expanded query string that adds context,
    specificity, or alternative phrasings to the original query.
    """

    expanded_query: str | None = Field(
        default=None,
        description="The expanded query string or None if expansion is not possible or needed.",
    )
