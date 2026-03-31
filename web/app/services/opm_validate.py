from __future__ import annotations

import logging

from pydantic import ValidationError  # noqa: F401 — re-exported for callers

from ..schemas.opm import OpmDiagram

logger = logging.getLogger(__name__)


def validate_diagram(data: dict) -> OpmDiagram:
    """
    Validate a candidate diagram dict.

    Returns:
        OpmDiagram on success

    Raises:
        pydantic.ValidationError on schema/field failures
        ValueError on graph-integrity failures
    """
    diagram = OpmDiagram.model_validate(data)
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
