from __future__ import annotations

from enum import Enum
from typing import Any, List

from pydantic import BaseModel, Field, model_validator


def dedupe_link_ids_raw(data: dict[str, Any]) -> dict[str, Any]:
    """Ensure each link has a unique id (LLMs often repeat the same id)."""
    data = {**data}
    links = data.get("links")
    if isinstance(links, tuple):
        links = list(links)
        data["links"] = links
    if not isinstance(links, list) or not links:
        return data
    seen: set[str] = set()
    new_links: list[Any] = []
    for i, lk in enumerate(links):
        if not isinstance(lk, dict):
            new_links.append(lk)
            continue
        lid = str(lk.get("id", "")).strip() or f"link-{i}"
        if lid in seen:
            base = lid
            n = 2
            while f"{base}-{n}" in seen:
                n += 1
            lid = f"{base}-{n}"
        seen.add(lid)
        new_links.append({**lk, "id": lid})
    return {**data, "links": new_links}


class NodeKind(str, Enum):
    object = "object"
    process = "process"
    state = "state"


class Relation(str, Enum):
    agent = "agent"
    instrument = "instrument"
    consumption = "consumption"
    result = "result"
    effect = "effect"
    aggregation = "aggregation"
    specialization = "specialization"
    characterization = "characterization"


class OpmNode(BaseModel):
    # Lowercase slug; allow underscores (LLMs often emit snake_case).
    id: str = Field(..., pattern=r"^[a-z0-9]+([_-][a-z0-9]+)*$")
    kind: NodeKind
    label: str = Field(..., min_length=1, max_length=160)


class OpmLink(BaseModel):
    id: str = Field(..., min_length=1)
    source: str
    target: str
    relation: Relation


class OpmDiagram(BaseModel):
    version: str = Field(default="1.0")
    nodes: List[OpmNode]
    links: List[OpmLink]

    @model_validator(mode="before")
    @classmethod
    def _dedupe_link_ids(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        return dedupe_link_ids_raw(data)

    @model_validator(mode="after")
    def check_graph_integrity(self) -> "OpmDiagram":
        node_ids = {n.id for n in self.nodes}

        # Rule: node IDs must be unique
        if len(node_ids) != len(self.nodes):
            raise ValueError("Duplicate node IDs detected")

        # Rule: link IDs must be unique
        link_ids = [lk.id for lk in self.links]
        if len(set(link_ids)) != len(link_ids):
            raise ValueError("Duplicate link IDs detected")

        # Rule: link endpoints must reference existing nodes
        for lk in self.links:
            if lk.source not in node_ids:
                raise ValueError(f"Link '{lk.id}' source '{lk.source}' not in nodes")
            if lk.target not in node_ids:
                raise ValueError(f"Link '{lk.id}' target '{lk.target}' not in nodes")

        kind_by_id = {n.id: n.kind for n in self.nodes}
        for lk in self.links:
            sk = kind_by_id[lk.source]
            tk = kind_by_id[lk.target]
            r = lk.relation
            if r in (Relation.agent, Relation.instrument, Relation.consumption):
                if sk != NodeKind.object or tk != NodeKind.process:
                    raise ValueError(
                        f"Link '{lk.id}': relation '{r.value}' must be object→process, "
                        f"got {sk.value}→{tk.value}"
                    )
            elif r == Relation.result:
                if sk != NodeKind.process or tk != NodeKind.object:
                    raise ValueError(
                        f"Link '{lk.id}': relation 'result' must be process→object, "
                        f"got {sk.value}→{tk.value} (use 'effect' for process→state)"
                    )
            elif r == Relation.effect:
                if sk != NodeKind.process or tk != NodeKind.state:
                    raise ValueError(
                        f"Link '{lk.id}': relation 'effect' must be process→state, "
                        f"got {sk.value}→{tk.value}"
                    )
            elif r == Relation.characterization:
                if sk != NodeKind.object or tk != NodeKind.state:
                    raise ValueError(
                        f"Link '{lk.id}': relation 'characterization' must be object→state, "
                        f"got {sk.value}→{tk.value}"
                    )
            elif r in (Relation.aggregation, Relation.specialization):
                if sk != NodeKind.object or tk != NodeKind.object:
                    raise ValueError(
                        f"Link '{lk.id}': relation '{r.value}' must be object→object, "
                        f"got {sk.value}→{tk.value}"
                    )

        return self
