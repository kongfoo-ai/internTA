from __future__ import annotations


def extract_opm_diagram(text: str) -> dict:
    """
    Phase 1 stub implementation.

    Ignores input text and returns a deterministic hardcoded diagram.
    This function will be replaced by an LLM-backed implementation in Phase 2.
    """
    return {
        "version": "1.0",
        "nodes": [
            {"id": "example-object", "kind": "object", "label": "example"}
        ],
        "links": [],
    }
