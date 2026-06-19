"""Abstract knowledge-graph store for ontology-grounded GraphRAG.

This is the graph counterpart of ``agentle.vector_stores.vector_store.VectorStore``:
a single backend instance addressed per tenant via a *graph name*, exposing async
primitives (with sync wrappers) for upserting ontology-typed triples, retrieving
anchored subgraphs, and running scoped Cypher.

Concrete backends (e.g. ``FalkorGraphStore``) implement the abstract async methods.
"""

from __future__ import annotations

import abc
import logging
import re
from collections.abc import Mapping, Sequence
from typing import Any

from rsb.coroutines.run_sync import run_sync

from agentle.graph_stores.models import GraphTriple, Subgraph

logger = logging.getLogger(__name__)

_IDENT_RE = re.compile(r"[^A-Za-z0-9_]")


class GraphStore(abc.ABC):
    """Backend-agnostic ontology-grounded graph store."""

    def __init__(self, *, graph_prefix: str = "kb_") -> None:
        self.graph_prefix = graph_prefix

    # -- naming / Cypher safety -------------------------------------------

    @staticmethod
    def sanitize_identifier(value: str, *, fallback: str = "Entity") -> str:
        """Make a string safe to interpolate as a Cypher label / relationship type.

        Cypher cannot parameterize labels or relationship types, so they must be
        validated and interpolated by hand. Non-identifier characters collapse to
        underscore; the result is guaranteed to start with a letter/underscore.
        """
        slug = _IDENT_RE.sub("_", str(value or "").strip()).strip("_")
        if not slug:
            return fallback
        if not re.match(r"^[A-Za-z_]", slug):
            slug = f"n_{slug}"
        return slug[:64]

    def graph_name_for(self, tenant: str) -> str:
        """Resolve a tenant identifier (e.g. subdomain) to its graph name."""
        slug = _IDENT_RE.sub("_", str(tenant or "").strip()) or "default"
        return f"{self.graph_prefix}{slug}"

    # -- async primitives (abstract) --------------------------------------

    @abc.abstractmethod
    async def health_check_async(self) -> bool:
        """Return True if the backend answers a trivial query."""
        ...

    @abc.abstractmethod
    async def query_async(
        self,
        cypher: str,
        *,
        graph_name: str,
        params: Mapping[str, Any] | None = None,
    ) -> Sequence[Mapping[str, Any]]:
        """Run a raw Cypher query, returning rows as header->value mappings."""
        ...

    @abc.abstractmethod
    async def upsert_triples_async(
        self,
        triples: Sequence[GraphTriple],
        *,
        graph_name: str,
    ) -> int:
        """Idempotently MERGE the given triples (nodes + edge). Returns count upserted."""
        ...

    @abc.abstractmethod
    async def neighborhood_async(
        self,
        *,
        graph_name: str,
        anchor_ids: Sequence[str],
        hops: int = 1,
        allowed_regra_ids: Sequence[str | int] | None = None,
        node_limit: int = 200,
    ) -> Subgraph:
        """Expand a bounded neighborhood around anchor nodes into a Subgraph.

        ``allowed_regra_ids`` scopes traversal to edges whose provenance ``regra_id``
        is permitted for the calling agent (RBAC / per-item visibility enforcement).
        """
        ...

    @abc.abstractmethod
    async def delete_regra_async(self, *, graph_name: str, regra_id: str | int) -> int:
        """Remove all edges (and orphaned nodes) sourced from a KB item. Returns deleted edges."""
        ...

    # -- sync wrappers -----------------------------------------------------

    def health_check(self) -> bool:
        return run_sync(self.health_check_async)

    def query(
        self,
        cypher: str,
        *,
        graph_name: str,
        params: Mapping[str, Any] | None = None,
    ) -> Sequence[Mapping[str, Any]]:
        return run_sync(self.query_async, cypher=cypher, graph_name=graph_name, params=params)

    def upsert_triples(self, triples: Sequence[GraphTriple], *, graph_name: str) -> int:
        return run_sync(self.upsert_triples_async, triples=triples, graph_name=graph_name)

    def neighborhood(
        self,
        *,
        graph_name: str,
        anchor_ids: Sequence[str],
        hops: int = 1,
        allowed_regra_ids: Sequence[str | int] | None = None,
        node_limit: int = 200,
    ) -> Subgraph:
        return run_sync(
            self.neighborhood_async,
            graph_name=graph_name,
            anchor_ids=anchor_ids,
            hops=hops,
            allowed_regra_ids=allowed_regra_ids,
            node_limit=node_limit,
        )

    def delete_regra(self, *, graph_name: str, regra_id: str | int) -> int:
        return run_sync(self.delete_regra_async, graph_name=graph_name, regra_id=regra_id)
