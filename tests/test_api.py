import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["DATABASE_URL"] = "sqlite:///./test_ct200.db"

import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.db import engine, Base

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
V1 = os.path.join(DATA, "ct200_manual.pdf")
V2 = os.path.join(DATA, "ct200_manual_v2.pdf")

client = TestClient(app)
state = {}


@pytest.fixture(autouse=True, scope="module")
def setup():
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
    for f in ["./test_ct200.db", "./llm_store/generations.json"]:
        if os.path.exists(f):
            os.remove(f)


def test_health():
    assert client.get("/health").status_code == 200


def test_ingest_v1():
    r = client.post("/api/v1/ingest/local", data={"pdf_path": V1, "document_name": "CT-200 Manual"})
    assert r.status_code == 200
    d = r.json()
    assert d["version_number"] == 1
    assert d["node_count"] > 20
    state["v1_ver_id"] = d["version_id"]
    state["doc_id"] = d["document_id"]


def test_list_docs():
    r = client.get("/api/v1/documents")
    assert r.status_code == 200
    assert len(r.json()) >= 1


def test_list_sections_v1():
    r = client.get("/api/v1/sections?version=1")
    d = r.json()
    assert d["version_number"] == 1
    assert len(d["sections"]) == 8  # 8 top-level sections
    state["sec_ids"] = [s["id"] for s in d["sections"]]


def test_get_node():
    nid = state["sec_ids"][0]  # section 1
    r = client.get(f"/api/v1/nodes/{nid}")
    assert r.status_code == 200
    d = r.json()
    assert d["heading"] == "Device Overview"
    assert len(d["children"]) == 2  # 1.1 and 1.2


def test_search():
    r = client.get("/api/v1/search?q=overpressure")
    assert r.json()["count"] > 0


def test_create_selection():
    # select section 4 (alarms) and its children
    nid = state["sec_ids"][3]  # section 4
    r = client.get(f"/api/v1/nodes/{nid}")
    kids = [c["id"] for c in r.json()["children"]]
    all_ids = [nid] + kids

    r = client.post("/api/v1/selections", json={"name": "Safety Tests", "node_ids": all_ids})
    assert r.status_code == 200
    d = r.json()
    state["sel_id"] = d["id"]
    assert d["name"] == "Safety Tests"
    assert len(d["items"]) == len(all_ids)


def test_generate():
    r = client.post("/api/v1/generate", json={"selection_id": state["sel_id"]})
    assert r.status_code == 200
    d = r.json()
    state["gen_id"] = d["id"]
    assert d["status"] in ("valid", "error")
    if d["status"] == "valid":
        assert len(d["test_cases"]) >= 1


def test_retrieve_generation():
    r = client.get(f"/api/v1/generations/{state['gen_id']}")
    assert r.status_code == 200


def test_retrieve_by_selection():
    r = client.get(f"/api/v1/generations/by-selection/{state['sel_id']}")
    assert r.status_code == 200
    assert len(r.json()) >= 1


def test_ingest_v2():
    r = client.post("/api/v1/ingest/local", data={"pdf_path": V2, "document_name": "CT-200 Manual"})
    assert r.status_code == 200
    d = r.json()
    assert d["version_number"] == 2
    state["v2_ver_id"] = d["version_id"]


def test_default_is_latest():
    r = client.get("/api/v1/sections")
    assert r.json()["version_number"] == 2


def test_v1_still_works():
    r = client.get("/api/v1/sections?version=1")
    assert r.json()["version_number"] == 1


def test_diff_changed_node():
    # battery life section changed between v1 and v2
    r = client.get("/api/v1/search?q=Battery+Life&version=1")
    hits = r.json()["results"]
    assert len(hits) > 0
    nid = hits[0]["id"]

    r = client.get(f"/api/v1/diff/{nid}")
    d = r.json()
    assert d["changed"] is True
    assert d["change_type"] == "modified"


def test_selection_staleness():
    r = client.get(f"/api/v1/selections/{state['sel_id']}")
    d = r.json()
    stale = [i for i in d["items"] if i["is_stale"]]
    assert len(stale) > 0, "some items should be stale after v2"


def test_generation_staleness():
    r = client.get(f"/api/v1/generations/{state['gen_id']}")
    d = r.json()
    assert "staleness" in d


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
