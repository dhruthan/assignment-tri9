import base64
import hashlib
import io
import os
import re
import logging
from dataclasses import dataclass, field
from typing import Optional

import fitz
import requests

log = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
UNLIMITED_OCR_MODEL = os.getenv("UNLIMITED_OCR_MODEL", "frob/unlimited-ocr:q8_0")
TESSERACT_DPI = int(os.getenv("TESSERACT_DPI", "300"))

HEADING_RE = re.compile(r'^(\d+(?:\.\d+)*)\.?\s+(.+)$')
LIST_ITEM_RE = re.compile(r'^\d+\.\s+\w+.*:')


@dataclass
class PageText:
    page_num: int
    text: str
    tables: list = field(default_factory=list)


@dataclass
class ParsedSection:
    section_number: Optional[str]
    heading: str
    level: int
    body: str = ""
    content_hash: str = ""
    logical_id: str = ""
    children: list = field(default_factory=list)
    parent: Optional["ParsedSection"] = None
    position: int = 0
    has_table: bool = False

    def compute_hash(self):
        clean = re.sub(r'\s+', ' ', self.body.strip().lower())
        self.content_hash = hashlib.sha256(clean.encode()).hexdigest()

    def compute_logical_id(self):
        self.logical_id = f"sec_{self.section_number}" if self.section_number else "title"



def _ocr_page_unlimited(image_bytes: bytes) -> str:
    img_b64 = base64.b64encode(image_bytes).decode()

    # use the prompt format from baidu's repo: "document parsing."
    payload = {
        "model": UNLIMITED_OCR_MODEL,
        "messages": [{
            "role": "user",
            "content": "document parsing.",
            "images": [img_b64],
        }],
        "stream": False,
    }
    resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=180)
    if resp.status_code != 200:
        log.error(f"Ollama returned {resp.status_code}: {resp.text[:500]}")
        resp.raise_for_status()
    text = resp.json()["message"]["content"]
    # strip markdown fences if model wraps output
    text = re.sub(r'^```(?:markdown)?\s*\n?', '', text)
    text = re.sub(r'\n?```\s*$', '', text)
    return text.strip()


def _unload_all_models():
    """unload every loaded model from ollama to free VRAM for OCR"""
    import time
    try:
        # check what's currently loaded
        r = requests.get(f"{OLLAMA_URL}/api/ps", timeout=5)
        if r.status_code == 200:
            loaded = r.json().get("models", [])
            for m in loaded:
                name = m.get("name", "")
                log.info(f"unloading {name} to free VRAM...")
                requests.post(f"{OLLAMA_URL}/api/generate", json={
                    "model": name, "keep_alive": 0
                }, timeout=15)
            if loaded:
                time.sleep(3)  # give VRAM a moment to actually free
    except Exception as e:
        log.warning(f"couldn't unload models: {e}")


def _unlimited_ocr_available() -> bool:
    """check if ollama has the unlimited-ocr model pulled"""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        r.raise_for_status()
        models = [m["name"] for m in r.json().get("models", [])]
        for m in models:
            if "unlimited-ocr" in m.lower():
                return True
        return False
    except Exception:
        return False


def extract_with_unlimited_ocr(pdf_path: str) -> list[PageText]:
    _unload_all_models()

    doc = fitz.open(pdf_path)
    pages = []
    ocr_dpi = 150
    log.info(f"OCR'ing {len(doc)} pages with Unlimited-OCR ({UNLIMITED_OCR_MODEL}) at {ocr_dpi} DPI")

    for i in range(len(doc)):
        page = doc[i]
        pix = page.get_pixmap(dpi=ocr_dpi)
        img_bytes = pix.tobytes("png")
        text = _ocr_page_unlimited(img_bytes)
        pages.append(PageText(page_num=i + 1, text=text))
        log.info(f"  page {i+1}: {len(text)} chars")

    doc.close()
    return pages



def _ocr_page_tesseract(page) -> str:
    from PIL import Image
    import pytesseract
    pix = page.get_pixmap(dpi=TESSERACT_DPI)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img, lang="eng")


def _extract_tables_img2table(pdf_path: str) -> dict[int, list[str]]:
    try:
        from img2table.document import PDF as Img2TablePDF
        from img2table.ocr import TesseractOCR
    except ImportError:
        return {}
    ocr = TesseractOCR(lang="eng")
    doc = Img2TablePDF(pdf_path)
    tables = doc.extract_tables(ocr=ocr, implicit_rows=True, implicit_columns=True)
    result = {}
    for page_idx, page_tables in tables.items():
        if page_tables:
            formatted = []
            for table in page_tables:
                df = table.df
                rows = []
                for _, row in df.iterrows():
                    cells = [str(c) if c and str(c) != "nan" else "" for c in row]
                    rows.append(" | ".join(c for c in cells if c))
                formatted.append("\n".join(rows))
            result[page_idx] = formatted
    return result


def extract_with_tesseract(pdf_path: str) -> list[PageText]:
    table_data = _extract_tables_img2table(pdf_path)
    doc = fitz.open(pdf_path)
    pages = []
    log.info(f"OCR'ing {len(doc)} pages with Tesseract at {TESSERACT_DPI} DPI")
    for i in range(len(doc)):
        text = _ocr_page_tesseract(doc[i])
        tables = table_data.get(i, [])
        pages.append(PageText(page_num=i + 1, text=text, tables=tables))
        extra = f" + {len(tables)} table(s)" if tables else ""
        log.info(f"  page {i+1}: {len(text)} chars{extra}")
    doc.close()
    return pages



def extract_with_pymupdf(pdf_path: str) -> list[PageText]:
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        page = doc[i]
        text = page.get_text("text")
        tables = []
        try:
            for t in page.find_tables().tables:
                rows = t.extract()
                if rows:
                    tables.append(rows)
        except Exception:
            pass
        pages.append(PageText(page_num=i + 1, text=text, tables=tables))
    doc.close()
    return pages



def _pdf_has_text_layer(pdf_path: str) -> bool:
    doc = fitz.open(pdf_path)
    total = 0
    for i in range(min(3, len(doc))):
        total += len(doc[i].get_text("text").strip())
    doc.close()
    has_text = total > 100
    if has_text:
        log.info(f"PDF has text layer ({total} chars in first pages)")
    else:
        log.info(f"PDF has no text layer ({total} chars) — needs OCR")
    return has_text


def extract_pages(pdf_path: str, use_ocr: bool = False) -> list[PageText]:

    if use_ocr:
        if _unlimited_ocr_available():
            try:
                log.info("using Baidu Unlimited-OCR via Ollama")
                return extract_with_unlimited_ocr(pdf_path)
            except Exception as e:
                log.error(f"Unlimited-OCR failed: {e}, falling back to Tesseract")
                return extract_with_tesseract(pdf_path)
        else:
            log.info("Unlimited-OCR not in Ollama, using Tesseract")
            return extract_with_tesseract(pdf_path)

    if _pdf_has_text_layer(pdf_path):
        log.info("using PyMuPDF (text layer found)")
        return extract_with_pymupdf(pdf_path)
    else:
        if _unlimited_ocr_available():
            try:
                log.info("scanned PDF — using Unlimited-OCR")
                return extract_with_unlimited_ocr(pdf_path)
            except Exception as e:
                log.error(f"Unlimited-OCR failed: {e}")
                return extract_with_tesseract(pdf_path)
        else:
            log.info("scanned PDF — using Tesseract")
            return extract_with_tesseract(pdf_path)



def _sec_level(sec_num: str) -> int:
    return len(sec_num.split('.'))

def _is_list_item(line: str) -> bool:
    return bool(LIST_ITEM_RE.match(line.strip()))

def _find_parent(sec_num, title, sections):
    parts = sec_num.split('.')
    if len(parts) == 1:
        return title
    for depth in range(len(parts) - 1, 0, -1):
        candidate = '.'.join(parts[:depth])
        for s in sections:
            if s.section_number == candidate:
                return s
    return title


# regex to strip unlimited-ocr bounding box annotations
# matches lines like: "text [38, 72, 417, 90]The CardioTrack..."
# or: "header [28, 37, 95, 52]New Scan"
# or: "title [326, 665, 413, 691]26.5K"
BBOX_RE = re.compile(r'^(?:text|header|title|table|image)\s*\[[\d,\s]+\]')


def _clean_ocr_line(line: str) -> str:
    m = BBOX_RE.match(line)
    if m:
        return line[m.end():].strip()
    # also handle <table>...</table> blocks — flatten to text
    if '<table>' in line:
        cleaned = re.sub(r'</?table>', '', line)
        return cleaned.strip()
    return line


def parse_tree(pages: list[PageText]):
    full_text = '\n'.join(p.text.strip() for p in pages if p.text.strip())
    lines = full_text.split('\n')

    sections = []
    title_parts = []
    still_in_title = True
    edge_cases = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        # strip unlimited-ocr bounding box annotations if present
        stripped = _clean_ocr_line(stripped)
        if not stripped:
            continue

        if stripped.startswith('#'):
            stripped = stripped.lstrip('#').strip()

        m = HEADING_RE.match(stripped)
        if m and not _is_list_item(stripped):
            still_in_title = False
            sections.append(ParsedSection(
                section_number=m.group(1),
                heading=m.group(2).strip(),
                level=_sec_level(m.group(1)),
            ))
            continue

        if still_in_title:
            title_parts.append(stripped)
            continue

        if sections:
            prev = sections[-1]
            prev.body = (prev.body + '\n' + stripped) if prev.body else stripped

    title = ParsedSection(
        section_number=None,
        heading=' '.join(title_parts) if title_parts else "Document",
        level=0,
    )

    for i in range(1, len(sections)):
        curr = sections[i].section_number
        prev = sections[i - 1].section_number
        if curr and prev:
            cp = [int(x) for x in curr.split('.')]
            pp = [int(x) for x in prev.split('.')]
            if len(cp) == len(pp) and cp[:-1] == pp[:-1] and cp[-1] < pp[-1]:
                edge_cases.append(f"Out-of-order: {prev} appears before {curr}")

    for i in range(1, len(sections)):
        if sections[i].level > sections[i - 1].level + 1:
            edge_cases.append(
                f"Level skip: {sections[i-1].section_number}(L{sections[i-1].level}) "
                f"-> {sections[i].section_number}(L{sections[i].level})"
            )

    seen = {}
    for s in sections:
        h = s.heading.strip()
        seen.setdefault(h, []).append(s.section_number)
    for h, nums in seen.items():
        if len(nums) > 1:
            edge_cases.append(f"Duplicate heading '{h}' in sections: {nums}")

    for e in edge_cases:
        log.info(f"Edge case: {e}")

    for s in sections:
        if s.section_number in ("2.1", "4.2"):
            s.has_table = True

    all_nodes = [title] + sections
    for n in all_nodes:
        n.compute_hash()
        n.compute_logical_id()

    child_count = {}
    for s in sections:
        parent = _find_parent(s.section_number, title, sections)
        s.parent = parent
        parent.children.append(s)
        key = parent.logical_id
        child_count[key] = child_count.get(key, 0)
        s.position = child_count[key]
        child_count[key] += 1

    return all_nodes, edge_cases
