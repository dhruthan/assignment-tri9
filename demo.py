#!/usr/bin/env python3
"""
demo script — hit every endpoint in order to show the full flow.

start the server first:
    cd ct200_project
    uvicorn app.main:app --reload

then run:
    python demo.py

if qianfan-ocr is slow (~10s/page) and you just want to test the API flow:
    USE_QIANFAN_OCR=0 uvicorn app.main:app --reload
    python demo.py
this skips OCR and uses PyMuPDF text extraction instead (instant).
"""

import json
import sys
import httpx

BASE = "http://localhost:8000"
c = httpx.Client(base_url=BASE, timeout=300)

USE_OCR = "--ocr" in sys.argv


def show(label, data):
    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    print(json.dumps(data, indent=2, default=str)[:2000])


def main():
    # health check
    r = c.get("/health")
    assert r.status_code == 200
    print("server is up")

    # ingest v1
    mode = "OCR (Unlimited-OCR / Tesseract)" if USE_OCR else "PyMuPDF"
    print(f"\n>>> ingesting v1 ({mode})...")
    r = c.post("/api/v1/ingest/local", data={"pdf_path": "data/ct200_manual.pdf", "use_ocr": str(USE_OCR).lower()})
    v1 = r.json()
    show("v1 ingested", v1)

    # list docs
    show("documents", c.get("/api/v1/documents").json())

    # browse sections
    r = c.get("/api/v1/sections", params={"version": 1})
    secs = r.json()
    show("v1 top-level sections", secs)

    # get node with children (section 4: alarms)
    sec4_id = secs["sections"][3]["id"]
    node = c.get(f"/api/v1/nodes/{sec4_id}").json()
    show(f"section 4: {node['heading']}", node)

    # search
    show("search 'overpressure'", c.get("/api/v1/search?q=overpressure").json())

    # create selection
    sel_ids = [sec4_id] + [kid["id"] for kid in node["children"]]
    r = c.post("/api/v1/selections", json={"name": "Safety Tests", "node_ids": sel_ids})
    sel = r.json()
    show("created selection", sel)
    sel_id = sel["id"]

    # generate test cases
    print("\n>>> generating test cases via ollama (or mock)...")
    r = c.post("/api/v1/generate", json={"selection_id": sel_id})
    gen = r.json()
    show("generated test cases", gen)
    gen_id = gen["id"]

    # retrieve generation
    show("retrieved generation", c.get(f"/api/v1/generations/{gen_id}").json())

    # ingest v2
    print(f"\n>>> ingesting v2 ({mode})...")
    r = c.post("/api/v1/ingest/local", data={"pdf_path": "data/ct200_manual_v2.pdf", "use_ocr": str(USE_OCR).lower()})
    show("v2 ingested", r.json())

    # default is now v2
    show("sections (default=latest=v2)", c.get("/api/v1/sections").json())

    # v1 still accessible
    print(f"v1 sections still accessible: {c.get('/api/v1/sections?version=1').json()['version_number']}")

    # diff a changed node
    r = c.get("/api/v1/search?q=Battery+Life&version=1")
    if r.json()["results"]:
        bid = r.json()["results"][0]["id"]
        show("diff: Battery Life v1 vs latest", c.get(f"/api/v1/diff/{bid}").json())

    # selection staleness
    show("selection staleness after v2", c.get(f"/api/v1/selections/{sel_id}").json())

    # generation staleness
    g = c.get(f"/api/v1/generations/{gen_id}").json()
    show("generation staleness", {"id": g["id"], "status": g.get("status"), "staleness": g.get("staleness")})



if __name__ == "__main__":
    main()
