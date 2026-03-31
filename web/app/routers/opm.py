from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError

from .. import db
from ..services.opm_extract import extract_opm_diagram
from ..services.opm_validate import validate_diagram


router = APIRouter(prefix="/opm", tags=["opm"])


@router.post("/extract")
def extract(payload: Dict[str, Any]) -> Dict[str, Any]:
    text = str(payload.get("text", "")).strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    note_id: Optional[int] = None
    if payload.get("save_note"):
        note_id = db.insert_note(text)

    raw_dict = extract_opm_diagram(text)

    try:
        validated = validate_diagram(raw_dict)
    except (ValidationError, ValueError) as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "opm_extraction_failed",
                "stage": "validation",
                "detail": str(exc),
            },
        )

    diagram_id = db.insert_opm_diagram(validated.model_dump(), note_id=note_id)
    return {"note_id": note_id, "diagram_id": diagram_id, "diagram": validated.model_dump()}


@router.get("")
def list_all() -> Dict[str, Any]:
    return {"diagrams": db.list_opm_diagrams()}


@router.get("/{diagram_id}")
def get_one(diagram_id: int) -> Dict[str, Any]:
    row = db.get_opm_diagram(diagram_id)
    if row is None:
        raise HTTPException(status_code=404, detail="diagram not found")
    return row
