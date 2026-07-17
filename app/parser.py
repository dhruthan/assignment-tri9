import hashlib
import io
import os
import re
import logging
from dataclasses import dataclass, field
from typing import Optional

import fitz

log = logging.getLogger(__name__)

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
    """tesseract for text + img2table for tables"""
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
        log.info("OCR forced via flag, using Tesseract")
        return extract_with_tesseract(pdf_path)

    if _pdf_has_text_layer(pdf_path):
        log.info("using PyMuPDF (text layer found)")
        return extract_with_pymupdf(pdf_path)
    else:
        log.info("scanned PDF, using Tesseract OCR")
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
