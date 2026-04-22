"""Shared test utilities for web app tests."""

from __future__ import annotations

import json
from typing import Any

# Valid OPM JSON that passes Phase 3 validation (used to mock the LLM).
VALID_OPM_LLM_JSON = json.dumps(
    {
        "version": "1.0",
        "nodes": [{"id": "example-object", "kind": "object", "label": "example"}],
        "links": [],
    }
)


def fake_opm_llm_success(system_prompt: str, user_prompt: str, **kwargs: Any) -> str:
    return VALID_OPM_LLM_JSON
