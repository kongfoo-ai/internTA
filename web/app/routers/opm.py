from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import ValidationError

from .. import db
from ..services.opm_validate import humanize_diagram_validation, validate_diagram


def _opm_error_stage(exc: BaseException) -> str:
    """Ensure 422 diagram validation failures are never labeled as generic 'llm' in the API."""
    stage = getattr(exc, "stage", "llm")
    code = getattr(exc, "status_code", None)
    if code == 422 and stage != "validation":
        det = str(getattr(exc, "detail", ""))
        if "OpmDiagram" in det or "Value error" in det:
            return "validation"
        if getattr(exc, "technical_detail", None):
            return "validation"
    return stage


router = APIRouter(prefix="/opm", tags=["opm"])


@router.post("/extract")
def extract(payload: Dict[str, Any]) -> Dict[str, Any]:
    from ..services.opm_extract import OPMExtractionError, extract_opm_diagram

    text = str(payload.get("text", "")).strip()
    if not text:
        raise HTTPException(status_code=400, detail="text is required")

    note_id: Optional[int] = None
    if payload.get("save_note"):
        note_id = db.insert_note(text)

    try:
        raw_dict = extract_opm_diagram(text)
    except OPMExtractionError as exc:
        err_body: Dict[str, Any] = {
            "error": "opm_extraction_failed",
            "stage": _opm_error_stage(exc),
            "detail": exc.detail,
            "raw_response": exc.raw_response,
        }
        td = getattr(exc, "technical_detail", None)
        if td:
            err_body["technical_detail"] = td
        raise HTTPException(status_code=exc.status_code, detail=err_body)

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
    return {"note_id": note_id, "diagram_id": diagram_id, "diagram": validated.model_dump()}


@router.get("")
def list_all(limit: Optional[int] = Query(None, ge=1, le=500)) -> Dict[str, Any]:
    return {"diagrams": db.list_opm_diagrams(limit=limit)}


@router.get("/{diagram_id}")
def get_one(diagram_id: int) -> Dict[str, Any]:
    row = db.get_opm_diagram(diagram_id)
    if row is None:
        raise HTTPException(status_code=404, detail="diagram not found")
    return row
