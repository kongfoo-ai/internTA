"""
Acceptance tests aligned with testcase.md:

1. Happy path: drug-label style text → valid diagram persisted as JSON.
2. Negative path: invalid diagram → 422 with clear validation/LLM stage and reason (no crash).
3. Persistence: data survives a fresh client / same DB file (simulated restart).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from .conftest import VALID_OPM_LLM_JSON, fake_opm_llm_success

# Minimal label line (testcase 1 — generic fictitious product wording).
TEXT_DRUG_LABEL_SNIPPET = (
    "INDICATIONS AND USAGE — EXAMPLIX (ficticia) injection is indicated as an adjunct to diet "
    "and increased physical activity for long-term condition management in adults meeting enrollment criteria."
)

# Dense mechanism text (testcase 2 — used as request body for negative-path tests; content mirrors
# testcase.md “complex drug mechanism” scenario; tests mock invalid extraction so this need not fail live LLM).
TEXT_COMPLEX_MECHANISM = """
The GLP-1 receptor is predominantly coupled to a **stimulatory G-protein**, known as **Gs**. Activation of this Gs protein initiates the cAMP-PKA signaling cascade. This is the canonical pathway responsible for most of GLP-1's key effects, especially in the pancreatic beta-cells which release insulin.

Here is a step-by-step explanation of the process:

1. **Binding:** The GLP-1 hormone, released from the gut after a meal, travels through the bloodstream and binds to its specific receptor (GLP-1R) on the surface of a target cell (e.g., a pancreatic beta-cell).
2. **G-Protein Activation:** This binding causes a conformational change in the GLP-1R, which in turn activates the associated Gs protein. The alpha subunit of the Gs protein (Gsα) releases its bound GDP and binds a new molecule of GTP, becoming active.
3. **Adenylyl Cyclase Activation:** The activated Gsα subunit then binds to and activates an enzyme called **adenylyl cyclase** (AC).
4. **cAMP Production:** Adenylyl cyclase converts ATP (adenosine triphosphate) into the crucial second messenger molecule, **cyclic AMP** (cAMP).
5. **PKA Activation:** The rise in intracellular cAMP levels leads to the activation of **Protein Kinase A** (PKA). cAMP binds to the regulatory subunits of PKA, causing them to detach from the catalytic subunits, thereby activating them.
6. **Downstream Cellular Effects:** The active PKA then phosphorylates (adds a phosphate group to) numerous target proteins within the cell. In a pancreatic beta-cell, this leads to:
    - Closure of ATP-sensitive potassium channels (KATP), which leads to membrane depolarization.
    - Opening of voltage-gated calcium channels, causing an influx of calcium ions (Ca²⁺).
    - Enhanced exocytosis (release) of insulin-containing granules from the cell.
    - Increased transcription of the proinsulin gene, leading to more insulin synthesis for future release.

This entire process coordinates receptor signaling, second messengers, and vesicle release to regulate insulin output.
""".strip()


@pytest.fixture()
def tmp_db(tmp_path: Path) -> Generator[Path, None, None]:
    db_file = tmp_path / "test.db"
    with patch("web.app.db.DB_PATH", db_file):
        from web.app import db as db_module

        db_module.init_db()
        yield db_file


@pytest.fixture()
def client(tmp_db: Path) -> Generator[TestClient, None, None]:
    sys.modules.setdefault("ollama", MagicMock())
    from web.app.main import app

    with patch(
        "web.app.services.opm_extract.call_llm", side_effect=fake_opm_llm_success
    ):
        with TestClient(app) as c:
            yield c


def test_tc1_happy_path_drug_label_text_stores_diagram_json(client: TestClient, tmp_db: Path):
    """Test case 1: extract once, JSON diagram stored in DB."""
    from web.app import db as db_module

    response = client.post(
        "/opm/extract",
        json={"text": TEXT_DRUG_LABEL_SNIPPET, "save_note": False},
    )
    assert response.status_code == 200, response.text
    data = response.json()
    assert "diagram_id" in data and "diagram" in data
    assert data["diagram"]["version"] == "1.0"
    assert isinstance(data["diagram"]["nodes"], list)
    assert isinstance(data["diagram"]["links"], list)

    row = db_module.get_opm_diagram(data["diagram_id"])
    assert row is not None
    assert row["diagram"] == data["diagram"]


def test_tc2_negative_path_validation_error_structured(client: TestClient, monkeypatch):
    """Test case 2: validation failure returns explicit stage and reason (no 500)."""
    bad = {
        "version": "1.0",
        "nodes": [{"id": "InvalidId", "kind": "object", "label": "X"}],
        "links": [],
    }
    monkeypatch.setattr("web.app.services.opm_extract.extract_opm_diagram", lambda text: bad)

    response = client.post(
        "/opm/extract",
        json={"text": TEXT_COMPLEX_MECHANISM, "save_note": False},
    )
    assert response.status_code == 422
    body = response.json()
    detail = body["detail"]
    assert detail["error"] == "opm_extraction_failed"
    assert detail["stage"] == "validation"
    assert "detail" in detail and detail["detail"]


def test_tc2_negative_path_llm_error_structured(client: TestClient):
    import web.app.services.opm_extract as opm_mod

    err = opm_mod.OPMExtractionError(502, "LLM unavailable", None)
    with patch.object(opm_mod, "call_llm", side_effect=err):
        response = client.post(
            "/opm/extract",
            json={"text": TEXT_COMPLEX_MECHANISM, "save_note": False},
        )
    assert response.status_code == 502
    detail = response.json()["detail"]
    assert detail["stage"] == "llm"
    assert detail["error"] == "opm_extraction_failed"
    assert detail["detail"] == "LLM unavailable"


def test_tc3_persistence_simulated_restart_new_client(tmp_db: Path):
    """Test case 3: same DB file — new TestClient still reads last diagram after 'restart'."""
    sys.modules.setdefault("ollama", MagicMock())
    for mod in list(sys.modules):
        if mod.startswith("web.app"):
            del sys.modules[mod]
    from web.app.main import app

    diagram_id: int
    payload: dict
    with patch(
        "web.app.services.opm_extract.call_llm", side_effect=fake_opm_llm_success
    ):
        with TestClient(app) as c1:
            r = c1.post("/opm/extract", json={"text": "persist check", "save_note": False})
            assert r.status_code == 200
            diagram_id = r.json()["diagram_id"]
            payload = r.json()["diagram"]

    for mod in list(sys.modules):
        if mod.startswith("web.app"):
            del sys.modules[mod]
    from web.app.main import app as app2

    with TestClient(app2) as c2:
        got = c2.get(f"/opm/{diagram_id}")
        assert got.status_code == 200
        assert got.json()["diagram"] == payload


def test_tc3_list_returns_latest_first(tmp_db: Path):
    sys.modules.setdefault("ollama", MagicMock())
    for mod in list(sys.modules):
        if mod.startswith("web.app"):
            del sys.modules[mod]
    from web.app.main import app

    with patch(
        "web.app.services.opm_extract.call_llm", side_effect=fake_opm_llm_success
    ):
        with TestClient(app) as c:
            c.post("/opm/extract", json={"text": "first", "save_note": False})
            r2 = c.post("/opm/extract", json={"text": "second", "save_note": False})
            second_id = r2.json()["diagram_id"]
            lst = c.get("/opm").json()["diagrams"]
    assert lst[0]["id"] == second_id
