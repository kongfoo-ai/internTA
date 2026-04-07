from __future__ import annotations

import hashlib
import os
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import APIRouter, Header, HTTPException

from .. import db
from ..services.llm import VLLMClientError, create_chat_completion, health_check

router = APIRouter(prefix="/chat", tags=["chat"])


def _require_api_token(authorization: Optional[str]) -> None:
    expected = os.getenv("API_TOKEN")
    if not expected:
        return
    if authorization != f"Bearer {expected}":
        raise HTTPException(status_code=401, detail="invalid API token")


def _resolve_user_id(
    payload: Dict[str, Any],
    authorization: Optional[str],
    x_user_id: Optional[str],
) -> str:
    user_id = x_user_id or payload.get("user_id")
    if user_id:
        return str(user_id)[:128]
    if authorization:
        digest = hashlib.sha256(authorization.encode("utf-8")).hexdigest()[:16]
        return f"token:{digest}"
    return "anonymous"


def _usage_row_to_dict(row: Any) -> Dict[str, Any]:
    return {
        "id": row["id"],
        "user_id": row["user_id"],
        "request_id": row["request_id"],
        "model": row["model"],
        "prompt_tokens": row["prompt_tokens"],
        "completion_tokens": row["completion_tokens"],
        "total_tokens": row["total_tokens"],
        "source": row["source"],
        "created_at": row["created_at"],
    }


@router.post("/completions")
def completions(
    payload: Dict[str, Any],
    authorization: Optional[str] = Header(default=None),
    x_user_id: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    _require_api_token(authorization)

    if payload.get("stream"):
        raise HTTPException(status_code=400, detail="streaming is not supported by this proxy yet")

    if not payload.get("messages"):
        raise HTTPException(status_code=400, detail="messages is required")

    user_id = _resolve_user_id(payload, authorization, x_user_id)
    temperature = payload.get("temperature", 0.8)
    max_tokens = payload.get("max_tokens", 8000)
    top_p = payload.get("top_p", 0.8)
    try:
        result = create_chat_completion(
            payload.get("messages"),
            model=payload.get("model"),
            temperature=float(temperature if temperature is not None else 0.8),
            max_tokens=int(max_tokens if max_tokens is not None else 8000),
            top_p=float(top_p if top_p is not None else 0.8),
            repetition_penalty=payload.get("repetition_penalty"),
        )
    except VLLMClientError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    request_id = result["id"] or str(uuid4())
    usage = result["usage"]
    usage_id = db.insert_token_usage(
        user_id=user_id,
        request_id=request_id,
        model=result["model"],
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        total_tokens=usage["total_tokens"],
        source="web",
    )

    return {
        "id": request_id,
        "object": "chat.completion",
        "model": result["model"],
        "choices": [
            {
                "index": 0,
                "message": {"role": result["role"], "content": result["content"]},
                "finish_reason": result["finish_reason"],
            }
        ],
        "usage": usage,
        "usage_id": usage_id,
        "user_id": user_id,
    }


@router.get("/usage")
def usage(
    user_id: Optional[str] = None,
    limit: int = 100,
    authorization: Optional[str] = Header(default=None),
) -> Dict[str, Any]:
    _require_api_token(authorization)
    summary = db.summarize_token_usage(user_id=user_id)
    rows = db.list_token_usage(user_id=user_id, limit=limit)
    return {
        "summary": {
            "request_count": summary["request_count"],
            "prompt_tokens": summary["prompt_tokens"],
            "completion_tokens": summary["completion_tokens"],
            "total_tokens": summary["total_tokens"],
        },
        "items": [_usage_row_to_dict(row) for row in rows],
    }


@router.get("/health")
def health(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    _require_api_token(authorization)
    return health_check()
