from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, Field, model_validator


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
    id: str = Field(..., pattern=r'^[a-z0-9]+(-[a-z0-9]+)*$')
    kind: NodeKind
    label: str = Field(..., min_length=1, max_length=80)


class OpmLink(BaseModel):
    id: str = Field(..., min_length=1)
    source: str
    target: str
    relation: Relation


class OpmDiagram(BaseModel):
    version: str = Field(default="1.0")
    nodes: List[OpmNode]
    links: List[OpmLink]

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

        return self
