"""
InternTA MCP server.

The MCP process delegates model inference to a vLLM OpenAI-compatible server.
This keeps GPU model serving, token accounting, and MCP tool exposure separated.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict

from dotenv import load_dotenv
from fastmcp import FastMCP

from web.app import db
from web.app.services.llm import VLLMClientError, create_chat_completion, health_check

load_dotenv()
db.init_db()

mcp = FastMCP(name="InternTA")


@mcp.tool()
def chat_completion(
    messages: str,
    temperature: float = 0.8,
    max_tokens: int = 8000,
    top_p: float = 0.8,
    user_id: str = "mcp",
    model: str | None = None,
    repetition_penalty: float | None = None,
) -> Dict[str, Any]:
    """
    Get a chat completion from the InternTA model through vLLM.

    messages must be a JSON array of OpenAI-style message objects.
    """
    try:
        message_list = json.loads(messages)
    except json.JSONDecodeError as exc:
        return {"error": f"Invalid messages JSON: {exc}", "content": "", "role": "assistant"}

    try:
        result = create_chat_completion(
            message_list,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
    except VLLMClientError as exc:
        return {"error": str(exc), "content": "", "role": "assistant"}

    usage = result["usage"]
    request_id = result["id"] or "mcp"
    db.insert_token_usage(
        user_id=user_id or "mcp",
        request_id=request_id,
        model=result["model"],
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        total_tokens=usage["total_tokens"],
        source="mcp",
    )

    return {
        "role": result["role"],
        "content": result["content"],
        "model": result["model"],
        "request_id": request_id,
        "usage": usage,
        "finish_reason": result["finish_reason"],
    }


@mcp.tool()
def internta_health() -> Dict[str, Any]:
    """Check whether the configured vLLM endpoint is reachable."""
    return health_check()


if __name__ == "__main__":
    transport = os.getenv("MCP_TRANSPORT")
    if transport:
        if transport == "http":
            transport = "streamable-http"
        run_kwargs: dict[str, Any] = {"transport": transport}
        if transport in {"http", "streamable-http", "sse"}:
            run_kwargs["host"] = os.getenv("MCP_HOST", "127.0.0.1")
            run_kwargs["port"] = int(os.getenv("MCP_PORT", "9000"))
        asyncio.run(mcp.run_async(**run_kwargs))
    else:
        mcp.run()
