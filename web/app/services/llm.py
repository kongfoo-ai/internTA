from __future__ import annotations

import os
from typing import Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


class VLLMClientError(RuntimeError):
    """Raised when the configured vLLM-compatible endpoint cannot serve a request."""


def get_vllm_base_url() -> str:
    return os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")


def get_vllm_api_key() -> str:
    return os.getenv("VLLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY"


def get_model_name(default: str = "internta") -> str:
    return os.getenv("VLLM_MODEL") or os.getenv("INTERTA_MODEL") or default


def get_client() -> OpenAI:
    return OpenAI(api_key=get_vllm_api_key(), base_url=get_vllm_base_url())


def normalize_messages(messages: Any) -> list[dict[str, str]]:
    if not isinstance(messages, list):
        raise VLLMClientError("messages must be a list")

    normalized: list[dict[str, str]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise VLLMClientError(f"messages[{index}] must be an object")
        role = str(message.get("role", "")).strip()
        if role not in {"system", "user", "assistant", "tool"}:
            raise VLLMClientError(f"messages[{index}].role is not valid")
        normalized.append({"role": role, "content": str(message.get("content", ""))})

    if not normalized:
        raise VLLMClientError("messages must not be empty")
    return normalized


def usage_to_dict(usage: Any) -> dict[str, int]:
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if hasattr(usage, "model_dump"):
        data = usage.model_dump()
    elif isinstance(usage, dict):
        data = usage
    else:
        data = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
        }
    return {
        "prompt_tokens": int(data.get("prompt_tokens") or 0),
        "completion_tokens": int(data.get("completion_tokens") or 0),
        "total_tokens": int(data.get("total_tokens") or 0),
    }


def create_chat_completion(
    messages: Any,
    *,
    model: Optional[str] = None,
    temperature: float = 0.8,
    max_tokens: int = 8000,
    top_p: float = 0.8,
    repetition_penalty: Optional[float] = None,
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": model or get_model_name(),
        "messages": normalize_messages(messages),
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "top_p": float(top_p),
    }
    if repetition_penalty is not None:
        request["extra_body"] = {"repetition_penalty": float(repetition_penalty)}

    try:
        response = get_client().chat.completions.create(**request)
    except Exception as exc:
        raise VLLMClientError(str(exc)) from exc

    if not response.choices:
        raise VLLMClientError("vLLM returned no choices")

    choice = response.choices[0]
    message = choice.message
    return {
        "id": getattr(response, "id", None),
        "model": getattr(response, "model", request["model"]),
        "role": getattr(message, "role", None) or "assistant",
        "content": getattr(message, "content", None) or "",
        "finish_reason": getattr(choice, "finish_reason", None),
        "usage": usage_to_dict(getattr(response, "usage", None)),
    }


def health_check() -> dict[str, Any]:
    try:
        models = get_client().models.list()
    except Exception as exc:
        return {
            "ready": False,
            "base_url": get_vllm_base_url(),
            "model": get_model_name(),
            "error": str(exc),
        }
    return {
        "ready": True,
        "base_url": get_vllm_base_url(),
        "model": get_model_name(),
        "models": [model.id for model in getattr(models, "data", [])],
    }
