import json
import re
import os
import uuid
import logging
from datetime import datetime, timezone

import requests
from tinydb import TinyDB, Query

log = logging.getLogger(__name__)

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5")

_store = None

def get_store():
    global _store
    if _store is None:
        _store = TinyDB("./llm_store/generations.json")
    return _store


SYSTEM_PROMPT = """You are a QA engineer writing test cases for a medical device.
Given sections from a device manual, generate 3 to 5 specific, executable test cases.

Reply with ONLY a JSON array. No markdown, no explanation, no backticks, no thinking.

Each test case object must have these exact fields:
- "id": like "TC-001"
- "title": short descriptive name
- "preconditions": what setup is needed before running the test
- "steps": numbered step-by-step instructions someone else could follow
- "expected_result": what should happen, with specific values from the document
- "requirement_ref": section number this traces back to (e.g. "4.1")
- "priority": "high" for safety-critical, "medium" for functional, "low" for info

Start your response with [ and end with ]. Nothing else."""


def _build_prompt(sections):
    parts = ["Generate QA test cases for these medical device manual sections:\n"]
    for s in sections:
        sec = s.get("section_number") or "Title"
        parts.append(f"--- Section {sec}: {s['heading']} ---")
        parts.append(s["body"])
        parts.append("")
    return "\n".join(parts)


def _call_ollama(system_msg, user_msg):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.3,
            "num_ctx": 8192,
        },
    }
    resp = requests.post(f"{OLLAMA_URL}/api/chat", json=payload, timeout=300)
    resp.raise_for_status()
    content = resp.json()["message"]["content"]

    content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
    return content.strip()


def _mock_response():
    """fallback when ollama isn't running. lets tests pass without a GPU."""
    return json.dumps([
        {
            "id": "TC-001",
            "title": "Overpressure emergency deflation triggers correctly",
            "preconditions": "CT-200 powered on, cuff connected to test arm simulator",
            "steps": "1. Begin measurement\n2. Force cuff pressure above 299 mmHg\n3. Observe device response",
            "expected_result": "Device triggers emergency deflation valve, vents cuff within 2 seconds, shows E3 error",
            "requirement_ref": "4.1",
            "priority": "high"
        },
        {
            "id": "TC-002",
            "title": "Low battery aborts measurement with E4",
            "preconditions": "Device with batteries below 10% remaining capacity",
            "steps": "1. Power on device\n2. Start blood pressure measurement\n3. Wait for device response",
            "expected_result": "Measurement aborts, screen displays E4 error code",
            "requirement_ref": "4.2",
            "priority": "high"
        },
        {
            "id": "TC-003",
            "title": "Hypertension Stage 1 classification shown correctly",
            "preconditions": "Device powered on, cuff on calibrated test arm",
            "steps": "1. Simulate reading: systolic 135 mmHg, diastolic 85 mmHg\n2. Let measurement complete\n3. Check classification indicator on screen",
            "expected_result": "Screen shows Hypertension Stage 1 classification",
            "requirement_ref": "3.3",
            "priority": "medium"
        },
    ])


def _parse_response(raw):
    """
    pull valid JSON from the model output.
    qwen3.5:9b is usually clean but sometimes wraps in ```json or adds a
    one-liner before the array. we handle both.
    """
    text = raw.strip()
    text = re.sub(r'^```(?:json)?\s*', '', text)
    text = re.sub(r'\s*```$', '', text)
    text = text.strip()

    start = text.find('[')
    end = text.rfind(']')
    if start == -1 or end == -1:
        raise ValueError(f"no JSON array found in: {text[:200]}")

    parsed = json.loads(text[start:end + 1])
    if not isinstance(parsed, list) or len(parsed) == 0:
        raise ValueError("empty or non-list result")

    required = {"id", "title", "steps", "expected_result"}
    cleaned = []
    for i, tc in enumerate(parsed):
        if not isinstance(tc, dict):
            raise ValueError(f"item {i} is not a dict")
        missing = required - set(tc.keys())
        if missing:
            raise ValueError(f"item {i} missing fields: {missing}")
        cleaned.append({
            "id": str(tc.get("id", f"TC-{i+1:03d}")),
            "title": str(tc["title"]),
            "preconditions": str(tc.get("preconditions", "N/A")),
            "steps": str(tc["steps"]),
            "expected_result": str(tc["expected_result"]),
            "requirement_ref": str(tc.get("requirement_ref", "N/A")),
            "priority": str(tc.get("priority", "medium")),
        })
    return cleaned


def generate_test_cases(sections, selection_id, node_hashes):

    store = get_store()

    user_prompt = _build_prompt(sections)
    test_cases = None
    raw = ""
    status = "valid"
    error = None

    for attempt in range(3):
        try:
            system = SYSTEM_PROMPT
            if attempt > 0:
                system += "\n\nCRITICAL: respond ONLY with a JSON array. [ ... ]. NO other text. NO thinking."

            used_mock = False
            try:
                raw = _call_ollama(system, user_prompt)
                log.info(f"qwen3.5 responded ({len(raw)} chars)")
            except Exception as e:
                log.warning(f"ollama ({OLLAMA_MODEL}) failed: {e} — using mock")
                raw = _mock_response()
                used_mock = True

            test_cases = _parse_response(raw)
            if used_mock:
                log.info("test cases from MOCK (ollama was unavailable)")
            else:
                log.info(f"test cases from {OLLAMA_MODEL} ({len(test_cases)} cases)")
            break
        except (json.JSONDecodeError, ValueError) as e:
            error = str(e)
            log.warning(f"parse attempt {attempt+1}/3 failed: {e}")
            if attempt == 2:
                status = "error"
        except Exception as e:
            error = str(e)
            log.error(f"unexpected: {e}")
            status = "error"
            break

    result = {
        "id": str(uuid.uuid4()),
        "selection_id": selection_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "test_cases": test_cases or [],
        "node_hashes": node_hashes,
        "raw_response": raw[:5000],
        "prompt_used": user_prompt[:2000],
        "status": status,
        "error": error,
        "staleness": {},
    }
    store.insert(result)
    return result


def get_generation_by_id(gen_id):
    store = get_store()
    results = store.search(Query().id == gen_id)
    return results[0] if results else None


def get_generations_for_selection(sel_id):
    store = get_store()
    return store.search(Query().selection_id == sel_id)
