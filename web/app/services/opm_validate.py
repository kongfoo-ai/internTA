from __future__ import annotations

import copy
import logging

from pydantic import ValidationError  # noqa: F401 — re-exported for callers

from ..schemas.opm import OpmDiagram

logger = logging.getLogger(__name__)


def humanize_diagram_validation(exc: BaseException) -> str:
    """
    Short, user-facing summary. Log `exc` at warning level for full Pydantic output.
    """
    if not isinstance(exc, ValidationError):
        return str(exc)

    errs = list(exc.errors())
    if not errs:
        return "The diagram data did not pass validation."

    n = len(errs)

    def is_node_id_pattern(e: dict[str, object]) -> bool:
        loc = e.get("loc") or ()
        return bool(
            e.get("type") == "string_pattern_mismatch"
            and len(loc) >= 3
            and loc[0] == "nodes"
            and isinstance(loc[1], int)
            and loc[2] == "id"
        )

    pid = sum(1 for e in errs if is_node_id_pattern(e))
    if pid == n:
        return (
            "One or more node IDs use a format we don't allow. "
            "Use lowercase letters and digits, with single hyphens or underscores between parts "
            "(for example glp1-receptor or glp1_receptor). Avoid spaces, dots, or uppercase."
        )
    if pid > 0:
        return (
            f"The diagram has {n} validation issue(s), including {pid} invalid node ID(s). "
            "IDs must be lowercase slugs: letters, digits, hyphens, and underscores only (no spaces)."
        )

    long_lbl = [
        e
        for e in errs
        if e.get("type") == "string_too_long" and "label" in (e.get("loc") or ())
    ]
    if long_lbl and len(long_lbl) == n:
        return (
            "One or more labels are too long. Shorten them or split the content into more nodes."
        )

    if all(e.get("type") == "enum" for e in errs):
        return (
            "One or more fields use a value that is not allowed "
            "(for example an unknown link relation or node kind). "
            "Check spelling against the allowed names."
        )

    msgs = " ".join(str(e.get("msg", "")) for e in errs)
    if "Duplicate node IDs" in msgs:
        return "Duplicate node IDs: every node must have a unique id."
    if "Duplicate link IDs" in msgs:
        return "Duplicate link IDs: every link must have a unique id."
    if "not in nodes" in msgs:
        return (
            "One or more links reference a missing node (source or target id does not match any node). "
            "Use the exact ids listed under \"nodes\" for every link."
        )

    if (
        "must be object→process" in msgs
        or "must be process→object" in msgs
        or "must be process→state" in msgs
        or "must be object→state" in msgs
        or "must be object→object" in msgs
    ):
        return (
            "One or more links break OPM rules: agent, instrument, and consumption go from an object to a process; "
            "result goes from process to object; effect from process to state; characterization from object to state; "
            "aggregation and specialization only between objects."
        )

    return (
        f"The diagram JSON has {n} validation issue(s). "
        "Typical fixes: valid node ids (lowercase, letters/digits/hyphens/underscores), "
        "link endpoints that match node ids, and allowed relation/kind values. Try extract again."
    )


def _normalize_link_endpoints_inplace(out: dict) -> None:
    """Align link source/target strings with node ids (e.g. 5 vs node5); coerce to str."""
    nodes = out.get("nodes")
    if not isinstance(nodes, list):
        return
    ids = set()
    for n in nodes:
        if isinstance(n, dict) and n.get("id") is not None:
            ids.add(str(n["id"]).strip())
    id_lower_to_canon = {e.lower(): e for e in ids}
    links = out.get("links")
    if not isinstance(links, list):
        return

    def canon(ref: object) -> str:
        s = str(ref).strip() if ref is not None else ""
        if s in ids:
            return s
        if s.lower() in id_lower_to_canon:
            return id_lower_to_canon[s.lower()]
        if s.isdigit():
            cand = f"node{s}"
            if cand in ids:
                return cand
            if cand.lower() in id_lower_to_canon:
                return id_lower_to_canon[cand.lower()]
        return s

    for lk in links:
        if not isinstance(lk, dict):
            continue
        if "source" in lk:
            lk["source"] = canon(lk.get("source"))
        if "target" in lk:
            lk["target"] = canon(lk.get("target"))


def _repair_common_llm_link_relations_inplace(out: dict) -> None:
    """
    Fix frequent LLM mistakes (only when source is already a process):

    - `result` with **process** source and **state** target → `effect`.
    - `effect` with **process** source and **object** target → `result`.

    Relation and node kinds are compared case-insensitively.
    """
    nodes = out.get("nodes")
    if not isinstance(nodes, list):
        return
    kind_by_id: dict[str, str] = {}
    for n in nodes:
        if isinstance(n, dict) and n.get("id") is not None:
            nid = str(n["id"]).strip()
            kind_by_id[nid] = str(n.get("kind", "")).strip().lower()

    links = out.get("links")
    if not isinstance(links, list):
        return
    for lk in links:
        if not isinstance(lk, dict):
            continue
        sid = str(lk.get("source", "")).strip()
        tid = str(lk.get("target", "")).strip()
        sk = kind_by_id.get(sid)
        tk = kind_by_id.get(tid)
        rel = str(lk.get("relation", "")).strip().lower()
        if rel == "result" and sk == "process" and tk == "state":
            lk["relation"] = "effect"
        elif rel == "effect" and sk == "process" and tk == "object":
            lk["relation"] = "result"


def repair_common_llm_link_relations(data: dict) -> dict:
    """Deep copy + normalize endpoints + relation repair (for tests and external use)."""
    out = copy.deepcopy(data)
    _normalize_link_endpoints_inplace(out)
    _repair_common_llm_link_relations_inplace(out)
    return out


def validate_diagram(data: dict) -> OpmDiagram:
    """
    Validate a candidate diagram dict.

    Returns:
        OpmDiagram on success

    Raises:
        pydantic.ValidationError on schema/field failures
        ValueError on graph-integrity failures
    """
    work = copy.deepcopy(data)
    _normalize_link_endpoints_inplace(work)
    _repair_common_llm_link_relations_inplace(work)
    diagram = OpmDiagram.model_validate(work)
    _warn_semantic(diagram)
    return diagram


def _warn_semantic(diagram: OpmDiagram) -> None:
    # Self-loops
    for lk in diagram.links:
        if lk.source == lk.target:
            logger.warning("Self-loop on node '%s' via link '%s'", lk.source, lk.id)

    # Duplicate (source, target, relation) triples
    seen: set[tuple] = set()
    for lk in diagram.links:
        key = (lk.source, lk.target, lk.relation)
        if key in seen:
            logger.warning(
                "Duplicate relation %s on (%s → %s)",
                lk.relation,
                lk.source,
                lk.target,
            )
        seen.add(key)
