"""FalkorDB-backed :class:`GraphStore` for ontology-grounded GraphRAG.

FalkorDB is a Redis module exposing OpenCypher. Multi-tenancy mirrors the Qdrant
collection-per-tenant model: one *graph* per tenant inside a shared instance,
addressed by name via :meth:`GraphStore.graph_name_for`.

The ``falkordb`` client is imported lazily (inside ``__init__``) so importing this
module never requires the dependency to be installed — the same pattern
``QdrantVectorStore`` uses for ``qdrant_client``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, override

from agentle.graph_stores.graph_store import GraphStore
from agentle.graph_stores.models import GraphEdge, GraphNode, GraphTriple, Subgraph

if TYPE_CHECKING:
    from falkordb.asyncio import FalkorDB

logger = logging.getLogger(__name__)

# Provenance carried on edges (and some nodes) so retrieved facts stay traceable.
PROVENANCE_KEYS = (
    "regra_id",
    "doc_id",
    "chunk_vector_id",
    "ontology_version",
    "confidence",
)

_PRIMITIVE = (str, int, float, bool)


def _coerce_props(props: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep only Cypher-safe primitive property values (FalkorDB rejects nested maps)."""
    out: dict[str, Any] = {}
    for key, value in dict(props or {}).items():
        if value is None:
            continue
        if isinstance(value, _PRIMITIVE):
            out[str(key)] = value
        else:
            out[str(key)] = str(value)
    return out


class FalkorGraphStore(GraphStore):
    """Knowledge-graph store backed by FalkorDB (async client)."""

    _client: FalkorDB

    def __init__(
        self,
        *,
        host: str,
        port: int = 6379,
        password: str | None = None,
        username: str | None = None,
        graph_prefix: str = "kb_",
    ) -> None:
        from falkordb.asyncio import FalkorDB

        super().__init__(graph_prefix=graph_prefix)
        self._client = FalkorDB(
            host=host, port=port, password=password, username=username
        )

    def _graph(self, graph_name: str) -> Any:
        return self._client.select_graph(graph_name)

    # -- decoding helpers --------------------------------------------------

    @staticmethod
    def _node_from_obj(obj: Any) -> GraphNode:
        props = dict(getattr(obj, "properties", {}) or {})
        labels = getattr(obj, "labels", None) or []
        label = (
            labels[0]
            if labels
            else str(getattr(obj, "label", "") or "Entity")
        )
        node_id = str(props.get("id") or getattr(obj, "id", "") or "")
        name = str(props.get("name") or "")
        return GraphNode(id=node_id, label=str(label), name=name, properties=props)

    @staticmethod
    def _edge_from_obj(obj: Any, source: GraphNode, target: GraphNode) -> GraphEdge:
        props = dict(getattr(obj, "properties", {}) or {})
        etype = str(
            getattr(obj, "relation", None) or getattr(obj, "type", "") or "RELATED_TO"
        )
        return GraphEdge(
            type=etype,
            source_id=source.id,
            target_id=target.id,
            properties=props,
        )

    @classmethod
    def _value_to_plain(cls, value: Any) -> Any:
        # Decode FalkorDB Node/Edge objects to plain dicts; pass primitives through.
        if hasattr(value, "properties") and (
            hasattr(value, "labels") or hasattr(value, "relation")
        ):
            return dict(getattr(value, "properties", {}) or {})
        return value

    def _rows_to_dicts(self, result: Any) -> list[dict[str, Any]]:
        header = getattr(result, "header", None) or []
        names = [self._header_name(col, idx) for idx, col in enumerate(header)]
        rows: list[dict[str, Any]] = []
        for row in getattr(result, "result_set", None) or []:
            entry: dict[str, Any] = {}
            for idx, value in enumerate(row):
                key = names[idx] if idx < len(names) else f"col{idx}"
                entry[key] = self._value_to_plain(value)
            rows.append(entry)
        return rows

    @staticmethod
    def _header_name(col: Any, idx: int) -> str:
        # FalkorDB headers are typically [type, name] pairs; be defensive across versions.
        if isinstance(col, (list, tuple)) and len(col) >= 2:
            return str(col[1])
        if isinstance(col, (bytes, bytearray)):
            return col.decode("utf-8", "ignore")
        return str(col) if col is not None else f"col{idx}"

    # -- primitives --------------------------------------------------------

    @override
    async def health_check_async(self) -> bool:
        try:
            graph = self._graph(f"{self.graph_prefix}healthcheck")
            await graph.query("RETURN 1")
            return True
        except Exception:
            logger.warning("FalkorDB health check failed", exc_info=True)
            return False

    @override
    async def query_async(
        self,
        cypher: str,
        *,
        graph_name: str,
        params: Mapping[str, Any] | None = None,
    ) -> Sequence[Mapping[str, Any]]:
        graph = self._graph(graph_name)
        result = await graph.query(cypher, params=dict(params or {}))
        return self._rows_to_dicts(result)

    @override
    async def upsert_triples_async(
        self,
        triples: Sequence[GraphTriple],
        *,
        graph_name: str,
    ) -> int:
        if not triples:
            return 0

        graph = self._graph(graph_name)

        # Labels/relationship types can't be parameterized, so group triples by
        # (subject_label, predicate, object_label) and UNWIND the rows per group —
        # one round-trip per distinct shape instead of one per triple.
        groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for triple in triples:
            subj_label = self.sanitize_identifier(triple.subject.label)
            obj_label = self.sanitize_identifier(triple.object.label)
            predicate = self.sanitize_identifier(triple.predicate, fallback="RELATED_TO")
            row = {
                "sid": str(triple.subject.id),
                "sname": str(triple.subject.name or ""),
                "sprops": _coerce_props(triple.subject.properties),
                "oid": str(triple.object.id),
                "oname": str(triple.object.name or ""),
                "oprops": _coerce_props(triple.object.properties),
                "rprops": _coerce_props(triple.properties),
            }
            groups.setdefault((subj_label, predicate, obj_label), []).append(row)

        upserted = 0
        for (subj_label, predicate, obj_label), rows in groups.items():
            cypher = (
                "UNWIND $rows AS row "
                f"MERGE (s:`{subj_label}` {{id: row.sid}}) "
                "ON CREATE SET s.name = row.sname "
                "SET s += row.sprops "
                f"MERGE (o:`{obj_label}` {{id: row.oid}}) "
                "ON CREATE SET o.name = row.oname "
                "SET o += row.oprops "
                f"MERGE (s)-[r:`{predicate}`]->(o) "
                "SET r += row.rprops"
            )
            await graph.query(cypher, params={"rows": rows})
            upserted += len(rows)

        return upserted

    @override
    async def neighborhood_async(
        self,
        *,
        graph_name: str,
        anchor_ids: Sequence[str],
        hops: int = 1,
        allowed_regra_ids: Sequence[str | int] | None = None,
        node_limit: int = 200,
    ) -> Subgraph:
        ids = [str(a) for a in anchor_ids if str(a)]
        if not ids:
            return Subgraph()

        graph = self._graph(graph_name)
        safe_hops = max(1, min(int(hops or 1), 4))
        safe_limit = max(1, min(int(node_limit or 200), 2000))

        params: dict[str, Any] = {"anchor_ids": ids}
        regra_filter = ""
        if allowed_regra_ids:
            params["allowed_regras"] = [str(r) for r in allowed_regra_ids]
            # Provenance is stored as strings; compare against the permitted set.
            regra_filter = "WHERE toString(rel.regra_id) IN $allowed_regras "

        expand_cypher = (
            "MATCH (a) WHERE a.id IN $anchor_ids "
            f"MATCH path = (a)-[*1..{safe_hops}]-(b) "
            "UNWIND relationships(path) AS rel "
            "WITH DISTINCT rel "
            f"{regra_filter}"
            "RETURN startNode(rel) AS s, rel AS r, endNode(rel) AS o "
            f"LIMIT {safe_limit}"
        )

        nodes: dict[str, GraphNode] = {}
        edges: dict[tuple[str, str, str], GraphEdge] = {}

        # Always include the anchor nodes, even when they have no permitted edges.
        anchors_result = await graph.query(
            "MATCH (a) WHERE a.id IN $anchor_ids RETURN a", params={"anchor_ids": ids}
        )
        for row in getattr(anchors_result, "result_set", None) or []:
            if row:
                node = self._node_from_obj(row[0])
                if node.id:
                    nodes[node.id] = node

        expand_result = await graph.query(expand_cypher, params=params)
        for row in getattr(expand_result, "result_set", None) or []:
            if len(row) < 3:
                continue
            source = self._node_from_obj(row[0])
            target = self._node_from_obj(row[2])
            if source.id:
                nodes[source.id] = source
            if target.id:
                nodes[target.id] = target
            edge = self._edge_from_obj(row[1], source, target)
            edges[(edge.source_id, edge.type, edge.target_id)] = edge

        return Subgraph(nodes=tuple(nodes.values()), edges=tuple(edges.values()))

    @override
    async def delete_regra_async(self, *, graph_name: str, regra_id: str | int) -> int:
        graph = self._graph(graph_name)
        params = {"regra": str(regra_id)}

        # Directed pattern so each relationship is counted once (undirected double-counts).
        count_result = await graph.query(
            "MATCH ()-[r]->() WHERE toString(r.regra_id) = $regra RETURN count(r) AS c",
            params=params,
        )
        deleted = 0
        rows = getattr(count_result, "result_set", None) or []
        if rows and rows[0]:
            try:
                deleted = int(rows[0][0])
            except (TypeError, ValueError):
                deleted = 0

        await graph.query(
            "MATCH ()-[r]-() WHERE toString(r.regra_id) = $regra DELETE r",
            params=params,
        )
        # Drop nodes that became orphaned and were sourced only from this KB item.
        await graph.query(
            "MATCH (n) WHERE toString(n.regra_id) = $regra AND NOT (n)--() DELETE n",
            params=params,
        )
        return deleted

    async def aclose(self) -> None:
        close = getattr(self._client, "aclose", None) or getattr(
            self._client, "close", None
        )
        if close is None:
            return
        try:
            result = close()
            if hasattr(result, "__await__"):
                await result
        except Exception:
            logger.debug("Error closing FalkorDB client", exc_info=True)
