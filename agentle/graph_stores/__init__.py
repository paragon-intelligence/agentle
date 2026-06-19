"""Ontology-grounded knowledge-graph stores for GraphRAG.

Graph counterpart of ``agentle.vector_stores``: a backend-agnostic ``GraphStore``
plus a FalkorDB implementation. Concrete clients are imported lazily, so importing
this package never pulls optional dependencies.
"""

from agentle.graph_stores.graph_store import GraphStore
from agentle.graph_stores.models import (
    GraphEdge,
    GraphNode,
    GraphTriple,
    Subgraph,
)

__all__ = [
    "GraphStore",
    "GraphNode",
    "GraphEdge",
    "GraphTriple",
    "Subgraph",
]
