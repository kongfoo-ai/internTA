# spec-valid.md

# OPM Validation Layer Spec

> Scope: Define and implement the validation layer that sits between the LLM response (or any candidate diagram dict) and DB insertion. This phase enforces the invariant that no invalid diagram is ever stored.

---

## 1. Overview

This phase introduces the validation layer for OPM diagrams.

By the end of this phase, the system should no longer persist arbitrary candidate diagrams directly from extraction. Instead, every diagram must pass schema and graph-integrity checks before being stored.

This phase assumes that upstream extraction already returns a Python `dict` or raises an exception. It does not parse raw JSON strings and does not call the LLM.

---

## 2. Goals

This phase has five goals:

1. Define a strict OPM schema using Pydantic.
2. Validate candidate diagram dicts against the schema.
3. Enforce graph-integrity rules that go beyond field typing.
4. Keep validation fully decoupled from DB and LLM internals.
5. Ensure that invalid diagrams are rejected before persistence.

---

## 3. Design Rationale

### 3.1 Validation is the source of truth for correctness

The LLM is not trusted to produce correct or valid diagrams.

Correctness is enforced here, in the validation layer, through:

* schema checks
* enum checks
* graph-integrity checks
* controlled warning behavior for non-fatal semantic issues

### 3.2 Why validation is a pure function

Validation should be testable in isolation.

For that reason:

* it receives a Python `dict`
* it returns a validated model
* it raises on failure
* it does not read/write the DB
* it does not call the LLM

This keeps the boundary clean and makes tests deterministic.

### 3.3 Upstream responsibility boundary

This phase assumes the LLM layer already handled:

* prompt construction
* model invocation
* timeout/retry
* JSON parsing

That means the validation layer receives:

* a Python `dict`, or
* nothing, because an exception was raised earlier

Validation does **not** handle raw string parsing.

---

## 4. Scope & Constraints

### 4.1 Scope

Included:

* OPM Pydantic models
* hard validation rules
* warning-only semantic checks
* router/service integration after extraction, before DB insert

Excluded:

* raw JSON parsing
* prompt logic
* model invocation
* DB schema changes
* frontend behavior

### 4.2 Constraints

* Validation is a pure function: `validate_diagram(data: dict) -> OpmDiagram`
* Input must be a Python dict
* Output is a validated Pydantic model
* Raises `ValidationError` or `ValueError` on failure
* No DB or LLM knowledge
* Existing `extract_action_items` pipeline remains untouched

---

## 5. Upstream Contract

The validation layer relies on the following upstream contract:

* `extract_opm_diagram(text)` must return a Python `dict` on success
* If upstream JSON parsing fails, the LLM layer must raise an exception before validation is called
* Validation must never be given:

  * raw JSON strings
  * `None`
  * partially parsed objects

This clarifies the phase boundary:

* **LLM layer** guarantees `dict-or-raise`
* **Validation layer** accepts `dict` and validates it

---

## 6. Pydantic Schema

### Module: `schemas/opm.py`

```python
from __future__ import annotations
from enum import Enum
from typing import Any, List
from pydantic import BaseModel, Field, model_validator


def dedupe_link_ids_raw(data: dict) -> dict:
    """Rename duplicate link ids (LLMs often repeat the same id)."""
    links = data.get("links")
    if not isinstance(links, list) or not links:
        return data
    seen: set[str] = set()
    new_links = []
    for i, lk in enumerate(links):
        if not isinstance(lk, dict):
            new_links.append(lk)
            continue
        lid = str(lk.get("id", "")).strip() or f"link-{i}"
        if lid in seen:
            n = 2
            while f"{lid}-{n}" in seen:
                n += 1
            lid = f"{lid}-{n}"
        seen.add(lid)
        new_links.append({**lk, "id": lid})
    return {**data, "links": new_links}


class NodeKind(str, Enum):
    object  = "object"
    process = "process"
    state   = "state"


class Relation(str, Enum):
    agent            = "agent"
    instrument       = "instrument"
    consumption      = "consumption"
    result           = "result"
    effect           = "effect"
    aggregation      = "aggregation"
    specialization   = "specialization"
    characterization = "characterization"


class OpmNode(BaseModel):
    # Lowercase slug; underscores and hyphens are both allowed (LLMs emit both).
    id:    str = Field(..., pattern=r'^[a-z0-9]+([_-][a-z0-9]+)*$')
    kind:  NodeKind
    label: str = Field(..., min_length=1, max_length=160)


class OpmLink(BaseModel):
    id:       str = Field(..., min_length=1)
    source:   str
    target:   str
    relation: Relation


class OpmDiagram(BaseModel):
    version: str = Field(default="1.0")
    nodes:   List[OpmNode]
    links:   List[OpmLink]

    @model_validator(mode="before")
    @classmethod
    def _dedupe_link_ids(cls, data: Any) -> Any:
        """Rename duplicate link ids before field validation (no hard reject)."""
        if not isinstance(data, dict):
            return data
        return dedupe_link_ids_raw(data)

    @model_validator(mode="after")
    def check_graph_integrity(self) -> "OpmDiagram":
        node_ids = {n.id for n in self.nodes}

        # Rule: node IDs must be unique
        if len(node_ids) != len(self.nodes):
            raise ValueError("Duplicate node IDs detected")

        # Rule: link IDs must be unique (duplicates are renamed by _dedupe_link_ids above)
        link_ids = [lk.id for lk in self.links]
        if len(set(link_ids)) != len(link_ids):
            raise ValueError("Duplicate link IDs detected")

        # Rule: link endpoints must reference existing nodes
        for lk in self.links:
            if lk.source not in node_ids:
                raise ValueError(f"Link '{lk.id}' source '{lk.source}' not in nodes")
            if lk.target not in node_ids:
                raise ValueError(f"Link '{lk.id}' target '{lk.target}' not in nodes")

        # Rule: link direction constraints per relation type
        kind_by_id = {n.id: n.kind for n in self.nodes}
        for lk in self.links:
            sk = kind_by_id[lk.source]
            tk = kind_by_id[lk.target]
            r  = lk.relation
            if r in (Relation.agent, Relation.instrument, Relation.consumption):
                if sk != NodeKind.object or tk != NodeKind.process:
                    raise ValueError(
                        f"Link '{lk.id}': '{r.value}' must be object→process, "
                        f"got {sk.value}→{tk.value}"
                    )
            elif r == Relation.result:
                if sk != NodeKind.process or tk != NodeKind.object:
                    raise ValueError(
                        f"Link '{lk.id}': 'result' must be process→object, "
                        f"got {sk.value}→{tk.value} (use 'effect' for process→state)"
                    )
            elif r == Relation.effect:
                if sk != NodeKind.process or tk != NodeKind.state:
                    raise ValueError(
                        f"Link '{lk.id}': 'effect' must be process→state, "
                        f"got {sk.value}→{tk.value}"
                    )
            elif r == Relation.characterization:
                if sk != NodeKind.object or tk != NodeKind.state:
                    raise ValueError(
                        f"Link '{lk.id}': 'characterization' must be object→state, "
                        f"got {sk.value}→{tk.value}"
                    )
            elif r in (Relation.aggregation, Relation.specialization):
                if sk != NodeKind.object or tk != NodeKind.object:
                    raise ValueError(
                        f"Link '{lk.id}': '{r.value}' must be object→object, "
                        f"got {sk.value}→{tk.value}"
                    )

        return self
```

---

## 7. Validation Rules

### 7.1 Mandatory rules (hard reject)

| #  | Rule                                                                   | Enforcement               |
| -- | ---------------------------------------------------------------------- | ------------------------- |
| 1  | Input is already parsed JSON object                                    | upstream contract         |
| 2  | `version` field present                                                | field default + type      |
| 3  | `nodes` and `links` are lists                                          | type annotation           |
| 4  | Each node has `id`, `kind`, `label`                                    | `OpmNode`                 |
| 5  | `kind` is valid enum                                                   | `NodeKind`                |
| 6  | `label` non-empty and ≤ 160 chars                                      | field constraints         |
| 7  | Node `id` matches slug pattern (lowercase, digits, hyphens/underscores) | field pattern            |
| 8  | Node IDs unique                                                        | model validator (after)   |
| 9  | Each link has `id`, `source`, `target`, `relation`                     | `OpmLink`                 |
| 10 | `relation` is valid enum                                               | `Relation`                |
| 11 | Link endpoints reference existing nodes                                | model validator (after)   |
| 12 | Duplicate link IDs renamed automatically (not rejected)                | model validator (before)  |
| 13 | `agent`/`instrument`/`consumption` must be object→process              | model validator (after)   |
| 14 | `result` must be process→object                                        | model validator (after)   |
| 15 | `effect` must be process→state                                         | model validator (after)   |
| 16 | `characterization` must be object→state                                | model validator (after)   |
| 17 | `aggregation`/`specialization` must be object→object                   | model validator (after)   |

### 7.2 Warning-only semantic checks

These do not reject the diagram:

| Check                                                   | Behavior    |
| ------------------------------------------------------- | ----------- |
| Duplicate relation on same `(source, target, relation)` | log warning |
| Self-loop (`source == target`)                          | log warning |

---

## 8. Public API

### Module: `services/opm_validate.py`

#### `validate_diagram`

```python
def validate_diagram(data: dict) -> OpmDiagram:
    """
    Normalize endpoints + repair common LLM mistakes + run Pydantic validation.

    Returns:
        OpmDiagram on success

    Raises:
        pydantic.ValidationError on schema/field failures
        ValueError on graph-integrity failures
    """
    work = copy.deepcopy(data)
    _normalize_link_endpoints_inplace(work)    # coerce int endpoints, case-normalize
    _repair_common_llm_link_relations_inplace(work)  # fix result↔effect, agent→instrument
    diagram = OpmDiagram.model_validate(work)
    _warn_semantic(diagram)
    return diagram
```

#### `repair_common_llm_link_relations`

Public deep-copy wrapper (used in tests and external callers):

```python
def repair_common_llm_link_relations(data: dict) -> dict:
    out = copy.deepcopy(data)
    _normalize_link_endpoints_inplace(out)
    _repair_common_llm_link_relations_inplace(out)
    return out
```

#### Internal repair functions

`_normalize_link_endpoints_inplace(out)` — coerces numeric link endpoints to `node{N}` strings, case-normalizes references to match node ids.

`_repair_common_llm_link_relations_inplace(out)` — applies three repairs:
1. `result` (process→state) → `effect`
2. `effect` (process→object) → `result`
3. `agent` from non-human source → `instrument` (checks source label against `_HUMAN_LABEL_HINTS`)

`_HUMAN_LABEL_HINTS` — frozenset of lowercase human-indicator terms:
`patient, clinician, physician, doctor, nurse, provider, user, operator, administrator, researcher, scientist, organization, company, team, committee, person, individual, staff, pharmacist, caregiver, subject, participant`

#### `humanize_diagram_validation`

Converts a Pydantic `ValidationError` or `ValueError` into a short, user-facing English message (no raw Pydantic field paths). Collapses groups of the same error type (e.g. many invalid node id patterns → one sentence).

#### `_warn_semantic`

Non-fatal checks — log warnings only, never reject:
- Self-loop (`source == target`)
- Duplicate `(source, target, relation)` triple

---

## 9. Integration Points

### 9.1 Integration boundary

The integration order after this phase is:

```text
Text → LLM extraction → JSON parse → validate_diagram → DB insert
```

DB insertion must happen only after validation succeeds.

### 9.2 Example integration in `routers/opm.py`

```python
from pydantic import ValidationError
from fastapi import HTTPException

from ..services.opm_extract import OPMExtractionError, extract_opm_diagram
from ..services.opm_validate import humanize_diagram_validation, validate_diagram

try:
    raw_dict = extract_opm_diagram(text)   # guaranteed: repaired dict or raises OPMExtractionError
except OPMExtractionError as exc:
    raise HTTPException(
        status_code=exc.status_code,
        detail={
            "error": "opm_extraction_failed",
            "stage": getattr(exc, "stage", "llm"),
            "detail": exc.detail,
            "raw_response": exc.raw_response,
        },
    )

# Second validation pass in the router (safety net — extract_opm_diagram already validates,
# but this catches any diagram injected via monkeypatch in tests or future code paths).
try:
    validated = validate_diagram(raw_dict)
except (ValidationError, ValueError) as exc:
    raise HTTPException(
        status_code=422,
        detail={
            "error": "opm_extraction_failed",
            "stage": "validation",
            "detail": humanize_diagram_validation(exc),
            "technical_detail": str(exc),
        },
    )

diagram_id = db.insert_opm_diagram(validated.model_dump(), note_id=note_id)
```

### 9.3 Core invariant

The DB insert only happens after `validate_diagram` succeeds.

This enforces the invariant:

> no invalid diagram is ever stored

---

## 10. Error Response Shape

Validation failures return `422 Unprocessable Entity`.

To align with the LLM phase, validation errors should use the same outer structure:

```json
{
  "error": "opm_extraction_failed",
  "stage": "validation",
  "detail": "validation error message"
}
```

Notes:

* `stage` must be `"validation"`
* validation failures do not require `raw_response`
* if the framework requires `detail=...`, the structured payload above should be placed inside `detail`

This keeps cross-phase error responses consistent.

---

## 11. Module Layout

```text
web/app/
  schemas/
    __init__.py
    opm.py              # OpmNode, OpmLink, OpmDiagram
  services/
    opm_validate.py     # validate_diagram() public function
```

---

## 12. Testing Plan

### 12.1 Schema / hard rejection tests

| Test                                          | What it checks                                                  |
| --------------------------------------------- | --------------------------------------------------------------- |
| `test_valid_diagram_passes`                   | valid farmer diagram returns `OpmDiagram`                       |
| `test_valid_diagram_version_preserved`        | `result.version == "1.0"`                                       |
| `test_empty_nodes_list_accepted`              | `nodes: [], links: []` passes                                   |
| `test_invalid_node_kind_rejected`             | `kind: "thing"` raises `ValidationError`                        |
| `test_invalid_relation_rejected`              | `relation: "causes"` raises `ValidationError`                   |
| `test_label_too_long_rejected`               | label > 160 chars raises `ValidationError`                      |
| `test_empty_label_rejected`                   | `label: ""` raises `ValidationError`                            |
| `test_invalid_node_id_with_space_rejected`    | `id: "farmer node"` raises `ValidationError`                    |
| `test_uppercase_node_id_rejected`             | `id: "UPPERCASE"` raises `ValidationError`                      |
| `test_snake_case_node_id_accepted`            | `id: "my_node_id"` passes (underscores allowed)                 |
| `test_duplicate_node_ids_rejected`            | duplicate node ids raise `ValidationError` or `ValueError`      |
| `test_duplicate_link_ids_renamed_automatically` | duplicate link ids are renamed (not rejected); e.g. `l1` → `l1-2` |
| `test_dangling_link_source_rejected`          | missing source node raises `ValidationError` or `ValueError`    |
| `test_dangling_link_target_rejected`          | missing target node raises `ValidationError` or `ValueError`    |

### 12.2 Link direction tests

| Test                                    | What it checks                                          |
| --------------------------------------- | ------------------------------------------------------- |
| `test_result_object_to_state_rejected`  | `result` object→state raises `ValueError` (use `effect`) |
| `test_effect_object_to_state_rejected`  | `effect` object→state raises `ValueError`               |
| `test_agent_object_to_state_rejected`   | `agent` non-human→state: repaired to `instrument`, then rejected for wrong direction |
| `test_instrument_object_to_object_rejected` | `instrument` object→object raises `ValueError`      |

### 12.3 Repair tests

| Test                                              | What it checks                                   |
| ------------------------------------------------- | ------------------------------------------------ |
| `test_repair_swaps_result_to_effect_when_target_is_state` | `result` (process→state) → `effect`    |
| `test_repair_swaps_effect_to_result_when_target_is_object` | `effect` (process→object) → `result`  |
| `test_normalize_numeric_link_endpoint_to_node_prefix` | source `5` → `"node5"`                    |
| `test_repair_result_to_effect_case_insensitive_relation` | `"Result"` (capital R) is still repaired |

### 12.4 Warning-only semantic tests

| Test                           | What it checks                                        |
| ------------------------------ | ----------------------------------------------------- |
| `test_self_loop_warns_not_rejects`   | self-loop logs warning, does not raise          |
| `test_duplicate_relation_warns`      | duplicate triple logs warning, does not raise   |

### 12.5 humanize tests

| Test                                          | What it checks                                           |
| --------------------------------------------- | -------------------------------------------------------- |
| `test_humanize_diagram_validation_collapses_many_id_pattern_errors` | multiple invalid id errors → one readable sentence |
| `test_humanize_opm_link_rule_errors`         | link direction error → "OPM rules…object to a process"   |

### 12.6 Integration tests

| Test                                         | What it checks                              |
| -------------------------------------------- | ------------------------------------------- |
| `test_valid_stub_passes_validation_and_stores` | valid extract → 200, `diagram.version == "1.0"` |
| `test_invalid_diagram_blocked_before_db_insert` | invalid extract → 422, `stage == "validation"` |
| `test_invalid_diagram_not_persisted`         | invalid extract → zero rows in `opm_diagrams` |

---

## 13. No New Dependencies

`pydantic` is already available through the FastAPI dependency tree.

No additional package changes are required for this phase.

---

## 14. Success Criteria

This phase is successful if:

* validation is fully decoupled from LLM and DB concerns
* valid diagrams are accepted
* invalid diagrams are rejected before persistence
* warning-only semantic issues do not block storage
* the system invariant is enforced

---

## 15. Known Limitations

This phase intentionally does not enforce:

* semantic truth of extracted diagrams
* domain correctness beyond schema/integrity
* stronger structural orientation rules for all relation types

Those concerns may be introduced in future work.
