from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from ..app.services.opm_extract import extract_opm_diagram


# ---------------------------------------------------------------------------
# Unit tests: stub extractor
# ---------------------------------------------------------------------------


def test_stub_returns_dict():
    result = extract_opm_diagram("any text")
    assert isinstance(result, dict)


def test_stub_version_field():
    result = extract_opm_diagram("any text")
    assert result["version"] == "1.0"


def test_stub_has_nodes_and_links():
    result = extract_opm_diagram("any text")
    assert "nodes" in result
    assert "links" in result


def test_stub_is_deterministic():
    assert extract_opm_diagram("foo") == extract_opm_diagram("bar")


# ---------------------------------------------------------------------------
# Unit tests: JSON round-trip
# ---------------------------------------------------------------------------


def test_payload_round_trip():
    original = {
        "version": "1.0",
        "nodes": [{"id": "example-object", "kind": "object", "label": "example"}],
        "links": [],
    }
    encoded = json.dumps(original)
    decoded = json.loads(encoded)
    assert decoded == original


def test_version_preserved_in_round_trip():
    payload = {"version": "1.0", "nodes": [], "links": []}
    assert json.loads(json.dumps(payload))["version"] == "1.0"


# ---------------------------------------------------------------------------
# Fixtures: in-memory DB for isolation
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Generator[Path, None, None]:
    db_file = tmp_path / "test.db"
    with patch("web.app.db.DB_PATH", db_file):
        from web.app import db as db_module
        db_module.init_db()
        yield db_file


# ---------------------------------------------------------------------------
# Unit tests: DB helpers
# ---------------------------------------------------------------------------


def test_insert_opm_diagram_returns_int(tmp_db: Path):
    from web.app import db as db_module
    payload = {"version": "1.0", "nodes": [], "links": []}
    diagram_id = db_module.insert_opm_diagram(payload)
    assert isinstance(diagram_id, int)


def test_insert_opm_diagram_multiple_distinct_ids(tmp_db: Path):
    from web.app import db as db_module
    payload = {"version": "1.0", "nodes": [], "links": []}
    id1 = db_module.insert_opm_diagram(payload)
    id2 = db_module.insert_opm_diagram(payload)
    assert id1 != id2


def test_get_opm_diagram_returns_parsed_dict(tmp_db: Path):
    from web.app import db as db_module
    payload = {"version": "1.0", "nodes": [{"id": "x"}], "links": []}
    diagram_id = db_module.insert_opm_diagram(payload)
    row = db_module.get_opm_diagram(diagram_id)
    assert row is not None
    assert isinstance(row["diagram"], dict)
    assert row["diagram"]["version"] == "1.0"


def test_get_opm_diagram_not_found_returns_none(tmp_db: Path):
    from web.app import db as db_module
    assert db_module.get_opm_diagram(99999) is None


def test_note_id_null_when_not_provided(tmp_db: Path):
    from web.app import db as db_module
    payload = {"version": "1.0", "nodes": [], "links": []}
    diagram_id = db_module.insert_opm_diagram(payload)
    row = db_module.get_opm_diagram(diagram_id)
    assert row is not None
    assert row["note_id"] is None


def test_list_opm_diagrams_returns_all(tmp_db: Path):
    from web.app import db as db_module
    payload = {"version": "1.0", "nodes": [], "links": []}
    db_module.insert_opm_diagram(payload)
    db_module.insert_opm_diagram(payload)
    diagrams = db_module.list_opm_diagrams()
    assert len(diagrams) == 2


def test_stored_payload_matches_inserted(tmp_db: Path):
    from web.app import db as db_module
    payload = {"version": "1.0", "nodes": [{"id": "n1"}], "links": []}
    diagram_id = db_module.insert_opm_diagram(payload)
    row = db_module.get_opm_diagram(diagram_id)
    assert row is not None
    assert row["diagram"] == payload


# ---------------------------------------------------------------------------
# Integration tests: API
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_db: Path) -> Generator[TestClient, None, None]:
    # ollama is not installed in this environment; stub it so the app can be imported
    sys.modules.setdefault("ollama", MagicMock())
    # Remove cached app import so the patched DB_PATH takes effect
    for mod in list(sys.modules):
        if mod.startswith("web.app"):
            del sys.modules[mod]
    from web.app.main import app
    with TestClient(app) as c:
        yield c


def test_post_extract_inserts_row(client: TestClient, tmp_db: Path):
    from web.app import db as db_module
    response = client.post("/opm/extract", json={"text": "some text", "save_note": False})
    assert response.status_code == 200
    data = response.json()
    assert "diagram_id" in data
    row = db_module.get_opm_diagram(data["diagram_id"])
    assert row is not None


def test_post_extract_response_shape(client: TestClient):
    response = client.post("/opm/extract", json={"text": "some text", "save_note": False})
    assert response.status_code == 200
    data = response.json()
    assert "note_id" in data
    assert "diagram_id" in data
    assert "diagram" in data
    diagram = data["diagram"]
    assert "version" in diagram
    assert "nodes" in diagram
    assert "links" in diagram


def test_post_extract_note_id_null_when_save_note_false(client: TestClient):
    response = client.post("/opm/extract", json={"text": "some text", "save_note": False})
    assert response.status_code == 200
    assert response.json()["note_id"] is None


def test_get_by_id_returns_same_payload(client: TestClient):
    post_resp = client.post("/opm/extract", json={"text": "some text", "save_note": False})
    diagram_id = post_resp.json()["diagram_id"]
    inserted_diagram = post_resp.json()["diagram"]

    get_resp = client.get(f"/opm/{diagram_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["diagram"] == inserted_diagram


def test_get_by_id_404_when_missing(client: TestClient):
    response = client.get("/opm/99999")
    assert response.status_code == 404


def test_get_list_returns_diagrams_key(client: TestClient):
    client.post("/opm/extract", json={"text": "a", "save_note": False})
    response = client.get("/opm")
    assert response.status_code == 200
    assert "diagrams" in response.json()


def test_multiple_inserts_distinct_ids(client: TestClient):
    r1 = client.post("/opm/extract", json={"text": "a", "save_note": False})
    r2 = client.post("/opm/extract", json={"text": "b", "save_note": False})
    assert r1.json()["diagram_id"] != r2.json()["diagram_id"]


def test_post_extract_missing_text_returns_400(client: TestClient):
    response = client.post("/opm/extract", json={"save_note": False})
    assert response.status_code == 400
