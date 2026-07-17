import os
import shutil
import difflib
import logging
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, HTTPException, Query as QParam, UploadFile, File, Form
from pydantic import BaseModel
from sqlalchemy.orm import Session as DBSession

from app.db import (
    get_db, init_db, new_id,
    Document, Version, Node, Selection, SelectionItem,
)
from app.parser import extract_pages, parse_tree
from app import llm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app):
    init_db()
    yield


app = FastAPI(
    title="CT-200 QA System",
    description="Parse the CT-200 manual, version it, generate QA test cases, track staleness.",
    version="1.0.0",
    lifespan=lifespan,
)



class SelectionCreate(BaseModel):
    name: str
    node_ids: list[str]
    version_id: Optional[str] = None

class GenerateRequest(BaseModel):
    selection_id: str


@app.get("/health")
def health():
    return {"status": "ok"}


def _do_ingest(db: DBSession, pdf_path: str, doc_name: str, filename: str, use_ocr: bool = False):
    """shared logic for both upload and local-path ingestion"""
    pages = extract_pages(pdf_path, use_ocr=use_ocr)
    all_nodes, edge_cases = parse_tree(pages)
    log.info(f"parsed {len(all_nodes)} nodes, {len(edge_cases)} edge cases")

    # find or create the document
    doc = db.query(Document).filter(Document.name == doc_name).first()
    if doc is None:
        doc = Document(id=new_id(), name=doc_name)
        db.add(doc)
        db.flush()
        ver_num = 1
    else:
        ver_num = db.query(Version).filter(Version.document_id == doc.id).count() + 1

    ver = Version(id=new_id(), document_id=doc.id, version_num=ver_num, filename=filename)
    db.add(ver)
    db.flush()

    # save nodes to db
    id_map = {}  # logical_id -> db id
    db_nodes = []
    for n in all_nodes:
        nid = new_id()
        id_map[n.logical_id] = nid

        parent_db_id = None
        if n.parent is not None:
            parent_db_id = id_map.get(n.parent.logical_id)

        db_node = Node(
            id=nid,
            version_id=ver.id,
            logical_id=n.logical_id,
            section_number=n.section_number,
            heading=n.heading,
            level=n.level,
            body=n.body,
            content_hash=n.content_hash,
            parent_id=parent_db_id,
            position=n.position,
            has_table=n.has_table,
        )
        db_nodes.append(db_node)

    db.add_all(db_nodes)
    db.commit()

    return {
        "document_id": doc.id,
        "version_id": ver.id,
        "version_number": ver_num,
        "node_count": len(db_nodes),
        "edge_cases": edge_cases,
    }


@app.post("/api/v1/ingest")
def ingest_upload(
    file: UploadFile = File(...),
    document_name: str = Form("CT-200 Manual"),
    use_ocr: bool = Form(False),
    db: DBSession = Depends(get_db),
):
    """upload a PDF file to ingest. set use_ocr=true to force Tesseract OCR."""
    tmp = f"/tmp/{file.filename}"
    with open(tmp, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        return _do_ingest(db, tmp, document_name, file.filename, use_ocr)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


@app.post("/api/v1/ingest/local")
def ingest_local(
    pdf_path: str = Form(...),
    document_name: str = Form("CT-200 Manual"),
    use_ocr: bool = Form(False),
    db: DBSession = Depends(get_db),
):
    """ingest from a local file path. set use_ocr=true to force Tesseract OCR."""
    if not os.path.exists(pdf_path):
        raise HTTPException(404, f"file not found: {pdf_path}")
    return _do_ingest(db, pdf_path, document_name, os.path.basename(pdf_path), use_ocr)



def _resolve_version(db, doc_name, version_num=None):
    """get a specific version or the latest one"""
    doc = db.query(Document).filter(Document.name == doc_name).first()
    if not doc:
        raise HTTPException(404, "document not found")

    if version_num:
        ver = db.query(Version).filter(
            Version.document_id == doc.id,
            Version.version_num == version_num,
        ).first()
    else:
        ver = db.query(Version).filter(
            Version.document_id == doc.id,
        ).order_by(Version.version_num.desc()).first()

    if not ver:
        raise HTTPException(404, "version not found")
    return ver


@app.get("/api/v1/documents")
def list_documents(db: DBSession = Depends(get_db)):
    docs = db.query(Document).all()
    result = []
    for doc in docs:
        versions = db.query(Version).filter(Version.document_id == doc.id).order_by(Version.version_num).all()
        result.append({
            "id": doc.id,
            "name": doc.name,
            "created_at": str(doc.created_at),
            "versions": [{"id": v.id, "version_num": v.version_num, "filename": v.filename} for v in versions],
            "latest_version": versions[-1].version_num if versions else None,
        })
    return result


@app.get("/api/v1/sections")
def list_sections(
    version: Optional[int] = QParam(None),
    document_name: str = QParam("CT-200 Manual"),
    db: DBSession = Depends(get_db),
):
    """top-level sections for a version (defaults to latest)"""
    ver = _resolve_version(db, document_name, version)

    title = db.query(Node).filter(Node.version_id == ver.id, Node.level == 0).first()
    if not title:
        return {"version_id": ver.id, "version_number": ver.version_num, "sections": []}

    kids = db.query(Node).filter(
        Node.version_id == ver.id, Node.parent_id == title.id
    ).order_by(Node.position).all()

    return {
        "version_id": ver.id,
        "version_number": ver.version_num,
        "sections": [_node_brief(db, n) for n in kids],
    }


@app.get("/api/v1/nodes/{node_id}")
def get_node(node_id: str, db: DBSession = Depends(get_db)):
    """get a node with its children"""
    node = db.query(Node).filter(Node.id == node_id).first()
    if not node:
        raise HTTPException(404, "node not found")

    kids = db.query(Node).filter(
        Node.parent_id == node.id, Node.version_id == node.version_id
    ).order_by(Node.position).all()

    return {
        "id": node.id,
        "logical_id": node.logical_id,
        "section_number": node.section_number,
        "heading": node.heading,
        "level": node.level,
        "body_text": node.body,
        "content_hash": node.content_hash,
        "parent_id": node.parent_id,
        "position": node.position,
        "has_table": node.has_table,
        "children": [_node_brief(db, c) for c in kids],
    }


@app.get("/api/v1/search")
def search_nodes(
    q: str = QParam(...),
    version: Optional[int] = QParam(None),
    search_body: bool = QParam(True),
    document_name: str = QParam("CT-200 Manual"),
    db: DBSession = Depends(get_db),
):
    ver = _resolve_version(db, document_name, version)
    query = db.query(Node).filter(Node.version_id == ver.id)

    if search_body:
        query = query.filter(
            (Node.heading.ilike(f"%{q}%")) | (Node.body.ilike(f"%{q}%"))
        )
    else:
        query = query.filter(Node.heading.ilike(f"%{q}%"))

    hits = query.all()
    return {
        "query": q,
        "version_number": ver.version_num,
        "results": [_node_brief(db, n) for n in hits],
        "count": len(hits),
    }


@app.get("/api/v1/diff/{node_id}")
def diff_node(
    node_id: str,
    target_version: Optional[int] = QParam(None),
    db: DBSession = Depends(get_db),
):
    #Check if a node changed between its version and another version
    node = db.query(Node).filter(Node.id == node_id).first()
    if not node:
        raise HTTPException(404, "node not found")

    cur_ver = db.query(Version).filter(Version.id == node.version_id).first()

    # figure out what we're comparing against
    if target_version:
        other_ver = db.query(Version).filter(
            Version.document_id == cur_ver.document_id,
            Version.version_num == target_version,
        ).first()
    else:
        other_ver = db.query(Version).filter(
            Version.document_id == cur_ver.document_id,
        ).order_by(Version.version_num.desc()).first()

    if not other_ver or other_ver.id == cur_ver.id:
        return {"logical_id": node.logical_id, "changed": False, "change_type": "unchanged"}

    # find same logical node in other version
    other_node = db.query(Node).filter(
        Node.version_id == other_ver.id, Node.logical_id == node.logical_id
    ).first()

    if not other_node:
        ct = "removed" if cur_ver.version_num < other_ver.version_num else "added"
        return {
            "logical_id": node.logical_id, "section_number": node.section_number,
            "heading": node.heading, "changed": True, "change_type": ct,
            "diff_summary": "Section doesn't exist in the other version",
        }

    if node.content_hash == other_node.content_hash:
        return {"logical_id": node.logical_id, "changed": False, "change_type": "unchanged"}

    # make a diff
    d = list(difflib.unified_diff(
        node.body.splitlines(), other_node.body.splitlines(),
        fromfile=f"v{cur_ver.version_num}", tofile=f"v{other_ver.version_num}",
        lineterm="",
    ))

    return {
        "logical_id": node.logical_id, "section_number": node.section_number,
        "heading": node.heading, "changed": True, "change_type": "modified",
        "diff_summary": "\n".join(d[:30]),
        "old_version": cur_ver.version_num, "new_version": other_ver.version_num,
    }



@app.post("/api/v1/selections")
def create_selection(body: SelectionCreate, db: DBSession = Depends(get_db)):
    """
    create a named selection. it's version-pinned: each item remembers
    exactly which version + content hash it was created against.
    """
    doc = db.query(Document).filter(Document.name == "CT-200 Manual").first()
    if not doc:
        raise HTTPException(400, "no document found")

    sel = Selection(id=new_id(), name=body.name, document_id=doc.id)
    db.add(sel)
    db.flush()

    items = []
    for nid in body.node_ids:
        node = db.query(Node).filter(Node.id == nid).first()
        if not node:
            db.rollback()
            raise HTTPException(400, f"node {nid} not found")

        items.append(SelectionItem(
            id=new_id(),
            selection_id=sel.id,
            node_id=node.id,
            version_id=body.version_id or node.version_id,
            pinned_hash=node.content_hash,
        ))

    db.add_all(items)
    db.commit()
    return _selection_detail(db, sel)


@app.get("/api/v1/selections")
def list_selections(db: DBSession = Depends(get_db)):
    sels = db.query(Selection).all()
    return [_selection_detail(db, s) for s in sels]


@app.get("/api/v1/selections/{sel_id}")
def get_selection(sel_id: str, db: DBSession = Depends(get_db)):
    sel = db.query(Selection).filter(Selection.id == sel_id).first()
    if not sel:
        raise HTTPException(404, "selection not found")
    return _selection_detail(db, sel)



@app.post("/api/v1/generate")
def generate(body: GenerateRequest, db: DBSession = Depends(get_db)):
    sel = db.query(Selection).filter(Selection.id == body.selection_id).first()
    if not sel:
        raise HTTPException(404, "selection not found")

    # gather section text
    sections = []
    hashes = {}
    for item in sel.items:
        node = item.node
        sections.append({
            "section_number": node.section_number,
            "heading": node.heading,
            "body": node.body,
            "logical_id": node.logical_id,
        })
        hashes[node.logical_id] = item.pinned_hash

    result = llm.generate_test_cases(sections, body.selection_id, hashes)

    # add staleness info
    result["staleness"] = _check_staleness(db, hashes, sel.document_id)
    return result


@app.get("/api/v1/generations/{gen_id}")
def get_generation(gen_id: str, db: DBSession = Depends(get_db)):
    gen = llm.get_generation_by_id(gen_id)
    if not gen:
        raise HTTPException(404, "generation not found")

    # add live staleness check
    if gen.get("selection_id"):
        sel = db.query(Selection).filter(Selection.id == gen["selection_id"]).first()
        if sel and gen.get("node_hashes"):
            gen["staleness"] = _check_staleness(db, gen["node_hashes"], sel.document_id)
    return gen


@app.get("/api/v1/generations/by-selection/{sel_id}")
def generations_by_selection(sel_id: str, db: DBSession = Depends(get_db)):
    sel = db.query(Selection).filter(Selection.id == sel_id).first()
    if not sel:
        raise HTTPException(404, "selection not found")

    gens = llm.get_generations_for_selection(sel_id)
    for g in gens:
        if g.get("node_hashes"):
            g["staleness"] = _check_staleness(db, g["node_hashes"], sel.document_id)
    return gens


@app.get("/api/v1/generations/by-node/{node_id}")
def generations_by_node(node_id: str, db: DBSession = Depends(get_db)):
    """find all generations that involved a particular node"""
    items = db.query(SelectionItem).filter(SelectionItem.node_id == node_id).all()
    sel_ids = {it.selection_id for it in items}
    gens = []
    for sid in sel_ids:
        gens.extend(llm.get_generations_for_selection(sid))
    return gens



def _check_staleness(db, node_hashes, document_id):

    latest = db.query(Version).filter(
        Version.document_id == document_id
    ).order_by(Version.version_num.desc()).first()

    if not latest:
        return {}

    latest_nodes = db.query(Node).filter(Node.version_id == latest.id).all()
    latest_map = {n.logical_id: n for n in latest_nodes}

    out = {}
    for lid, old_hash in node_hashes.items():
        current = latest_map.get(lid)
        if not current:
            out[lid] = {"stale": True, "reason": "section removed in latest version"}
        elif current.content_hash == old_hash:
            out[lid] = {"stale": False, "reason": "unchanged"}
        else:
            # find original to make a diff
            original = db.query(Node).filter(
                Node.logical_id == lid, Node.content_hash == old_hash
            ).first()
            diff_text = None
            if original:
                d = list(difflib.unified_diff(
                    original.body.splitlines(), current.body.splitlines(),
                    fromfile="original", tofile="current", lineterm=""
                ))
                diff_text = "\n".join(d[:20])

            out[lid] = {"stale": True, "reason": "content changed", "diff": diff_text}

    return out



def _node_brief(db, node):
    has_kids = db.query(Node).filter(
        Node.parent_id == node.id, Node.version_id == node.version_id
    ).count() > 0
    return {
        "id": node.id,
        "logical_id": node.logical_id,
        "section_number": node.section_number,
        "heading": node.heading,
        "level": node.level,
        "content_hash": node.content_hash,
        "has_children": has_kids,
    }


def _selection_detail(db, sel):
    latest = db.query(Version).filter(
        Version.document_id == sel.document_id
    ).order_by(Version.version_num.desc()).first()

    items_out = []
    for item in sel.items:
        node = item.node
        stale = False
        if latest and latest.id != item.version_id:
            latest_node = db.query(Node).filter(
                Node.version_id == latest.id, Node.logical_id == node.logical_id
            ).first()
            if latest_node:
                stale = latest_node.content_hash != item.pinned_hash
            else:
                stale = True  # gone in latest

        items_out.append({
            "node_id": item.node_id,
            "version_id": item.version_id,
            "logical_id": node.logical_id,
            "section_number": node.section_number,
            "heading": node.heading,
            "pinned_hash": item.pinned_hash,
            "is_stale": stale,
        })

    return {
        "id": sel.id,
        "name": sel.name,
        "document_id": sel.document_id,
        "created_at": str(sel.created_at),
        "items": items_out,
    }
