"""Lightweight data models for ontology-grounded knowledge graphs.

These mirror the role that ``Chunk`` plays for vector stores: the minimal,
backend-agnostic shapes a GraphRAG pipeline passes around (nodes, edges, the
triples that are upserted, and the subgraphs that are retrieved).

Provenance lives in node/edge ``properties`` (e.g. ``regra_id``, ``doc_id``,
``chunk_vector_id``, ``ontology_version``, ``confidence``) so retrieved facts can
always be traced back to source documents/chunks and verified.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GraphNode:
    """An ontology-typed entity in a tenant knowledge graph.

    ``label`` is the ontology class (used as the Cypher node label). ``id`` must be
    a stable, ontology-normalized key so the same real-world entity merges across
    documents instead of duplicating.
    """

    id: str
    label: str
    name: str = ""
    properties: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphEdge:
    """An ontology-typed relation between two entities, carrying provenance."""

    type: str
    source_id: str
    target_id: str
    properties: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphTriple:
    """A subject-predicate-object fact ready to be upserted into a graph."""

    subject: GraphNode
    predicate: str
    object: GraphNode
    properties: Mapping[str, Any] = field(default_factory=dict)
    """Provenance / metadata for the edge (regra_id, doc_id, chunk_vector_id, ...)."""

    def to_edge(self) -> GraphEdge:
        return GraphEdge(
            type=self.predicate,
            source_id=self.subject.id,
            target_id=self.object.id,
            properties=self.properties,
        )


@dataclass(frozen=True)
class Subgraph:
    """A connected (or anchored) set of nodes and edges returned by retrieval."""

    nodes: Sequence[GraphNode] = field(default_factory=tuple)
    edges: Sequence[GraphEdge] = field(default_factory=tuple)

    @property
    def is_empty(self) -> bool:
        return not self.nodes and not self.edges
