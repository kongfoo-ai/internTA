from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from web.app.services.opm_validate import validate_diagram
from web.app.schemas.opm import OpmDiagram


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VALID_DIAGRAM: dict = {
    "version": "1.0",
    "nodes": [
        {"id": "farmer", "kind": "object", "label": "Farmer"},
        {"id": "grow", "kind": "process", "label": "Grow"},
        {"id": "crop", "kind": "object", "label": "Crop"},
    ],
    "links": [
        {"id": "l1", "source": "farmer", "target": "grow", "relation": "agent"},
        {"id": "l2", "source": "grow", "target": "crop", "relation": "result"},
    ],
}


def _diagram(**overrides) -> dict:
    """Return a copy of VALID_DIAGRAM with top-level fields overridden."""
    return {**VALID_DIAGRAM, **overrides}


# ---------------------------------------------------------------------------
# Unit tests: valid diagram
# ---------------------------------------------------------------------------


def test_valid_diagram_passes():
    result = validate_diagram(VALID_DIAGRAM)
    assert isinstance(result, OpmDiagram)


def test_valid_diagram_version_preserved():
    result = validate_diagram(VALID_DIAGRAM)
    assert result.version == "1.0"


def test_empty_nodes_list_accepted():
    result = validate_diagram({"version": "1.0", "nodes": [], "links": []})
    assert isinstance(result, OpmDiagram)
    assert result.nodes == []
    assert result.links == []


# ---------------------------------------------------------------------------
# Unit tests: hard rejections
# ---------------------------------------------------------------------------


def test_invalid_node_kind_rejected():
    bad = _diagram(nodes=[{"id": "x", "kind": "thing", "label": "X"}])
    with pytest.raises(ValidationError):
        validate_diagram(bad)


def test_invalid_relation_rejected():
    bad = _diagram(
        links=[{"id": "l1", "source": "farmer", "target": "grow", "relation": "causes"}]
    )
    with pytest.raises(ValidationError):
        validate_diagram(bad)


def test_label_too_long_rejected():
    long_label = "a" * 81
    bad = _diagram(nodes=[{"id": "farmer", "kind": "object", "label": long_label}])
    with pytest.raises(ValidationError):
        validate_diagram(bad)


def test_empty_label_rejected():
    bad = _diagram(nodes=[{"id": "farmer", "kind": "object", "label": ""}])
    with pytest.raises(ValidationError):
        validate_diagram(bad)


def test_non_kebab_node_id_rejected():
    bad = _diagram(nodes=[{"id": "MyNode", "kind": "object", "label": "My Node"}])
    with pytest.raises(ValidationError):
        validate_diagram(bad)


def test_uppercase_node_id_rejected():
    bad = _diagram(nodes=[{"id": "UPPERCASE", "kind": "object", "label": "X"}])
    with pytest.raises(ValidationError):
        validate_diagram(bad)


def test_duplicate_node_ids_rejected():
    bad = _diagram(
        nodes=[
            {"id": "farmer", "kind": "object", "label": "Farmer"},
            {"id": "farmer", "kind": "process", "label": "Farmer 2"},
        ]
    )
    with pytest.raises((ValidationError, ValueError)):
        validate_diagram(bad)


def test_duplicate_link_ids_rejected():
    bad = _diagram(
        links=[
            {"id": "l1", "source": "farmer", "target": "grow", "relation": "agent"},
            {"id": "l1", "source": "grow", "target": "crop", "relation": "result"},
        ]
    )
    with pytest.raises((ValidationError, ValueError)):
        validate_diagram(bad)


def test_dangling_link_source_rejected():
    bad = _diagram(
        links=[{"id": "l1", "source": "ghost", "target": "grow", "relation": "agent"}]
    )
    with pytest.raises((ValidationError, ValueError)):
        validate_diagram(bad)


def test_dangling_link_target_rejected():
    bad = _diagram(
        links=[{"id": "l1", "source": "farmer", "target": "ghost", "relation": "agent"}]
    )
    with pytest.raises((ValidationError, ValueError)):
        validate_diagram(bad)


# ---------------------------------------------------------------------------
# Unit tests: warning-only semantic checks
# ---------------------------------------------------------------------------


def test_self_loop_warns_not_rejects(caplog):
    diagram = {
        "version": "1.0",
        "nodes": [{"id": "node-a", "kind": "object", "label": "A"}],
        "links": [
            {"id": "l1", "source": "node-a", "target": "node-a", "relation": "effect"}
        ],
    }
    with caplog.at_level(logging.WARNING, logger="web.app.services.opm_validate"):
        result = validate_diagram(diagram)
    assert isinstance(result, OpmDiagram)
    assert any("Self-loop" in r.message for r in caplog.records)


def test_duplicate_relation_warns(caplog):
    diagram = {
        "version": "1.0",
        "nodes": [
            {"id": "node-a", "kind": "object", "label": "A"},
            {"id": "node-b", "kind": "process", "label": "B"},
        ],
        "links": [
            {"id": "l1", "source": "node-a", "target": "node-b", "relation": "agent"},
            {"id": "l2", "source": "node-a", "target": "node-b", "relation": "agent"},
        ],
    }
    with caplog.at_level(logging.WARNING, logger="web.app.services.opm_validate"):
        result = validate_diagram(diagram)
    assert isinstance(result, OpmDiagram)
    assert any("Duplicate relation" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Integration tests: router behaviour
# ---------------------------------------------------------------------------


@pytest.fixture()
def client() -> Generator:
    sys.modules.setdefault("ollama", MagicMock())
    for mod in list(sys.modules):
        if mod.startswith("web.app"):
            del sys.modules[mod]
    from fastapi.testclient import TestClient
    from web.app.main import app
    with TestClient(app) as c:
        yield c


def test_valid_stub_passes_validation_and_stores(client, tmp_path):
    with patch("web.app.db.DB_PATH", tmp_path / "test.db"):
        from web.app import db as db_module
        db_module.init_db()
        response = client.post("/opm/extract", json={"text": "some text", "save_note": False})
    assert response.status_code == 200
    data = response.json()
    assert data["diagram"]["version"] == "1.0"


def test_invalid_diagram_blocked_before_db_insert(client, tmp_path, monkeypatch):
    """An invalid dict from extraction must be rejected with 422, not stored."""
    bad_diagram = {"version": "1.0", "nodes": [{"id": "BadID", "kind": "object", "label": "X"}], "links": []}
    # Patch the name as imported in the router module
    monkeypatch.setattr("web.app.routers.opm.extract_opm_diagram", lambda text: bad_diagram)

    with patch("web.app.db.DB_PATH", tmp_path / "test.db"):
        from web.app import db as db_module
        db_module.init_db()
        response = client.post("/opm/extract", json={"text": "some text", "save_note": False})

    assert response.status_code == 422
    body = response.json()
    assert body["detail"]["stage"] == "validation"
    assert body["detail"]["error"] == "opm_extraction_failed"


def test_invalid_diagram_not_persisted(client, tmp_path, monkeypatch):
    """After a validation failure, no row should appear in opm_diagrams."""
    bad_diagram = {"version": "1.0", "nodes": [{"id": "Bad", "kind": "object", "label": "X"}], "links": []}
    monkeypatch.setattr("web.app.routers.opm.extract_opm_diagram", lambda text: bad_diagram)

    with patch("web.app.db.DB_PATH", tmp_path / "test.db"):
        from web.app import db as db_module
        db_module.init_db()
        client.post("/opm/extract", json={"text": "some text", "save_note": False})
        diagrams = db_module.list_opm_diagrams()

    assert diagrams == []
