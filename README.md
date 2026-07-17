## Tech Stack

- **FastAPI** + **Pydantic** + **SQLAlchemy** + **SQLite** - API, validation, document tree storage
- **TinyDB** - JSON-based NoSQL store for LLM-generated test cases
- **Baidu Unlimited-OCR** via Ollama - primary OCR engine for scanned PDFs
- **Tesseract + img2table** - fallback OCR
- **Qwen 3.5 (9B)** via Ollama - local LLM for test case generation

## Setup

```bash
# clone
git clone https://github.com/dhruthan/assignment-tri9.git
cd assignment-tri9

# python deps
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# install tesseract (fallback OCR)
sudo apt install tesseract-ocr

# pull ollama models
ollama pull frob/unlimited-ocr:q8_0    # 4GB - Baidu Unlimited-OCR
ollama pull qwen3.5                    # 6.6GB
```

## Environment Variables (all optional)

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server URL |
| `DATABASE_URL` | `sqlite:///./ct200.db` | SQLite database path |

## Running

```bash
# start the server
uvicorn app.main:app --reload

python demo.py

# run with OCR forced (Unlimited-OCR -> Tesseract fallback)
python demo.py --ocr
```

## Running Tests

```bash
# all 37 tests
pytest tests/ -v

# parser edge case tests only
pytest tests/test_parser.py -v

# API integration tests only
pytest tests/test_api.py -v
```

## V1 → V2 Re-ingestion Flow

This is the core versioning + staleness flow:

```bash
# clean state
rm -f ct200.db llm_store/generations.json

# start server
uvicorn app.main:app --reload

# in another terminal — the demo does it automatically:
python demo.py
```

What `demo.py` does step by step:
1. Ingests `data/ct200_manual.pdf` as version 1 (28 nodes)
2. Lists top-level sections, gets node details
3. Creates a named selection on Section 4 (Alarms & Safety)
4. Generates QA test cases via qwen3.5
5. Ingests `data/ct200_manual_v2.pdf` as version 2 (29 nodes — 5.3 added)
6. Verifies v1 is still accessible
7. Shows diff on Battery Life section (300→250 cycles, 15%→10%)
8. Shows staleness: sec_4.2 stale (E3 2s→1.5s, E6 added), sec_4.3 stale (E1–E5→E1–E6)

You can also trigger re-ingestion manually:

```bash
# ingest v1
curl -X POST http://localhost:8000/api/v1/ingest/local \
  -F "pdf_path=data/ct200_manual.pdf"

# ingest v2 (creates version 2 under same document)
curl -X POST http://localhost:8000/api/v1/ingest/local \
  -F "pdf_path=data/ct200_manual_v2.pdf"

# force OCR on any ingestion
curl -X POST http://localhost:8000/api/v1/ingest/local \
  -F "pdf_path=data/ct200_manual.pdf" -F "use_ocr=true"
```

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/ingest/local` | Ingest PDF from local path |
| `POST` | `/api/v1/ingest` | Ingest uploaded PDF |
| `GET` | `/api/v1/documents` | List all documents with versions |
| `GET` | `/api/v1/sections?version=N` | Top-level sections (default: latest) |
| `GET` | `/api/v1/nodes/{id}` | Node detail with children |
| `GET` | `/api/v1/search?q=term` | Search headings and body text |
| `GET` | `/api/v1/diff/{node_id}` | Diff node across versions |
| `POST` | `/api/v1/selections` | Create named, version-pinned selection |
| `GET` | `/api/v1/selections/{id}` | Get selection with staleness info |
| `POST` | `/api/v1/generate` | Generate test cases for a selection |
| `GET` | `/api/v1/generations/{id}` | Retrieve generation with staleness |
| `GET` | `/api/v1/generations/by-selection/{id}` | All generations for a selection |
| `GET` | `/api/v1/generations/by-node/{id}` | All generations involving a node |
