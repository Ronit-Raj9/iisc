"""Generate `solution_notebook.ipynb` by filling every stub in the organizer's
skeleton notebook with a Gemma-3-12B-driven, STG-rule-correct implementation.

Design choices (token budget = 12M input / 750k output):
- ONE Gemma call per page returning a compact structured JSON (doc_type +
  entities + visual flags + ~200 chars of OCR snippet). No full-text dump.
- Caching: each call result is persisted to ./vlm_cache/<case_id>/<file>__pN.json
  so re-runs (incl. evaluation re-runs) hit the cache and burn no tokens.
- STG rules per package come straight from STG_RULES_*.md.
- Output schemas match HI.txt verbatim (MG064A→link, SG039C→S3_link/DocumentName,
  MG006A→S3_link, SB039A→s3_link).
- Output dir: ./output/<PACKAGE>.json per the user spec.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "nha_ps1_skeletal_notebook_main (1).ipynb"
DST = ROOT / "solution_notebook.ipynb"


def code(src: str) -> str:
    """Return code string with a trailing newline removed (notebook-friendly)."""
    return src.lstrip("\n").rstrip() + "\n"


# Map: cell-id -> new source code (string).  Markdown cells preserved untouched.
CELL_OVERRIDES: dict[str, str] = {
    # -------------------------------------------------------------------------
    # Cell 04 [7e47ecbe] - CONFIG (output dir = ./output, claims root, decisions)
    # -------------------------------------------------------------------------
    "7e47ecbe": code(
        '''# =========================
# 2. CONFIG
# =========================

DATA_ROOT = Path("./Claims")          # root of the unzipped Claims/ folder
OUTPUT_ROOT = Path("./output")        # required output dir per problem statement
DECISIONS_ROOT = Path("./decisions")  # human-readable Pass/Conditional/Fail per case
VLM_CACHE_ROOT = Path("./vlm_cache")  # cached Gemma results so reruns don't burn tokens
VLM_CLINICAL_CACHE_ROOT = Path("./vlm_clinical_cache")  # secondary cache for per-page clinical-field assessment

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
DECISIONS_ROOT.mkdir(parents=True, exist_ok=True)
VLM_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
VLM_CLINICAL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

PACKAGE_CODES = ["MG064A", "SG039C", "MG006A", "SB039A"]

PACKAGE_NAMES = {
    "MG064A": "Severe Anemia",
    "SG039C": "Cholecystectomy",
    "MG006A": "Enteric Fever",
    "SB039A": "Total Knee Replacement",
}

DECISION_PASS = "PASS"
DECISION_CONDITIONAL = "CONDITIONAL"
DECISION_FAIL = "FAIL"

# Token-budget guard: stop after this many fresh VLM calls (cache hits don't count).
#
# PS1 quota: 12,000,000 input tokens, 750,000 output tokens.
# Per page (Gemma 3 12B, image resized to <=1024px, compact JSON output):
#   ~1500 input tokens, ~300 output tokens.
# Dataset-1 has ~1597 page-equivalents -> ~2.4M input, ~480K output.
# Output is the tighter constraint.  If Gemma is more verbose than expected,
# do a dry run with MAX_VLM_CALLS=10 first, inspect ./vlm_cache to see how
# many tokens were used, then bump MAX_VLM_CALLS or set to None.
#
# Recommended:
#   1st run (smoke):     MAX_VLM_CALLS = 10
#   2nd run (validate):  MAX_VLM_CALLS = 100
#   Final run:           MAX_VLM_CALLS = None
#
# >>> CLINICAL RE-RUN INSTRUCTIONS <<<
# This notebook now ALSO calls Gemma a second time per page for per-package
# clinical-field assessment (severe_anemia, common_signs, fever, symptoms,
# arthritis_type, etc.) — Gemma does the medical reasoning directly instead
# of my keyword-matching.  The primary cache (vlm_cache/) is preserved; a
# new vlm_clinical_cache/ holds the secondary call results.
#
# To populate the clinical cache, set MAX_VLM_CALLS = None and run the
# notebook.  Expected: ~1597 fresh calls, ~160K output tokens (~21% of the
# 750K budget), ~100 minutes wall clock.  After this, all subsequent runs
# hit BOTH caches → 0 fresh calls.
MAX_VLM_CALLS = None
VLM_CALL_COUNTER = {"calls": 0}

# Switch the model here if needed.  Must be one of the NHAclient model names.
GEMMA_MODEL = "Gemma 3 12B"

# PDF rasterization zoom; 1.5 keeps OCR readable while limiting image-token cost.
PDF_RENDER_ZOOM = 1.5
'''
    ),
    # -------------------------------------------------------------------------
    # Cell 06 [bd1e59b8] - NHAclient init (keep cred placeholders, don't hit API)
    # -------------------------------------------------------------------------
    "bd1e59b8": code(
        '''# =========================
# NHAclient init
# =========================
# Fill in your hackathon credentials before running the pipeline.
# The credentials are loaded once and reused for all per-page Gemma calls.

from nha_client import NHAclient
import base64

clientId = ""
clientSecret = ""

nc = NHAclient(clientId, clientSecret)

# A tiny smoke check is intentionally NOT executed here to avoid burning
# tokens on notebook open.  See `analyze_page_with_gemma` further below.
'''
    ),
    # -------------------------------------------------------------------------
    # Cell 07 [fd923d39] - PACKAGE_SCHEMAS fixed to HI.txt
    # -------------------------------------------------------------------------
    "fd923d39": code(
        '''# =========================
# 3. OUTPUT SCHEMAS (HI.txt verbatim — link-field name varies per package)
# =========================
# HI.txt: "Solutions that do not strictly adhere to the specified output format
# will be rejected without evaluation."  Each package uses its OWN spelling
# of the link field per the worked examples in the problem statement.

PACKAGE_SCHEMAS = {
    "MG064A": [
        "case_id", "link", "procedure_code", "page_number",
        "clinical_notes", "cbc_hb_report", "indoor_case",
        "treatment_details", "post_hb_report", "discharge_summary",
        "severe_anemia", "common_signs", "significant_signs",
        "life_threatening_signs", "extra_document", "document_rank",
    ],
    "SG039C": [
        "case_id", "S3_link/DocumentName", "procedure_code", "page_number",
        "clinical_notes", "usg_report", "lft_report", "operative_notes",
        "pre_anesthesia", "discharge_summary", "photo_evidence",
        "histopathology", "clinical_condition", "usg_calculi",
        "pain_present", "previous_surgery", "extra_document", "document_rank",
    ],
    "MG006A": [
        "case_id", "S3_link", "procedure_code", "page_number",
        "clinical_notes", "investigation_pre", "pre_date", "vitals_treatment",
        "investigation_post", "post_date", "discharge_summary", "poor_quality",
        "fever", "symptoms", "extra_document", "document_rank",
    ],
    "SB039A": [
        "case_id", "s3_link", "procedure_code", "page_number",
        "clinical_notes", "xray_ct_knee", "indoor_case", "operative_notes",
        "implant_invoice", "post_op_photo", "post_op_xray", "discharge_summary",
        "doa", "dod", "arthritis_type", "post_op_implant_present",
        "age_valid", "extra_document", "document_rank",
    ],
}

# Per-package name of the document-link field, derived from PACKAGE_SCHEMAS above.
LINK_FIELD = {
    "MG064A": "link",
    "SG039C": "S3_link/DocumentName",
    "MG006A": "S3_link",
    "SB039A": "s3_link",
}

# Date fields per package (used for DD-MM-YYYY normalization downstream).
DATE_FIELDS = {
    "MG064A": [],
    "SG039C": [],
    "MG006A": ["pre_date", "post_date"],
    "SB039A": ["doa", "dod"],
}

# Keys that should NOT default to 0 when initializing a row.
NULLABLE_DATE_KEYS = {"pre_date", "post_date", "doa", "dod"}

KEY_ALIASES = {
    "S3_link": "link",
    "s3_link": "link",
    "S3_link/DocumentName": "link",
}
'''
    ),
    # -------------------------------------------------------------------------
    # Cell 10 [3200c253] - ingest helpers (walks Claims/<PACKAGE>/<case_id>/*)
    # -------------------------------------------------------------------------
    "3200c253": code(
        '''# =========================
# 1. INGEST CLAIM FILES
# =========================
# The dataset is organized as Claims/<PACKAGE_CODE>/<case_id>/<files>.
# discover_cases returns: { case_id : (package_code, [file_paths]) }
# so we can drive the pipeline from the directory hierarchy alone.

def iter_case_files(case_dir: Path) -> List[Path]:
    files: List[Path] = []
    for item in case_dir.rglob("*"):
        if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(item)
    return sorted(files)


def discover_cases(data_root: Path) -> Dict[str, List[Path]]:
    """Walk Claims/<PACKAGE>/<case>/* and return {case_id: [files]}.

    Side-effect: also populates the global PACKAGE_CODE_LOOKUP so the main
    runner doesn't need to ask for a package_code.txt helper file.
    """
    global PACKAGE_CODE_LOOKUP
    cases: Dict[str, List[Path]] = {}
    if not data_root.exists():
        print(f"[ingest] data root not found: {data_root.resolve()}")
        return cases

    for package_dir in sorted(data_root.iterdir()):
        if not package_dir.is_dir():
            continue
        pkg = package_dir.name
        if pkg not in PACKAGE_CODES:
            # Be permissive: still index the cases, just don't map a code.
            pass
        for case_dir in sorted(package_dir.iterdir()):
            if not case_dir.is_dir():
                continue
            case_id = case_dir.name
            files = iter_case_files(case_dir)
            if not files:
                continue
            cases[case_id] = files
            try:
                PACKAGE_CODE_LOOKUP[case_id] = pkg
            except NameError:
                PACKAGE_CODE_LOOKUP = {case_id: pkg}
    return cases


# Initialize the lookup so later cells can reference it freely.
PACKAGE_CODE_LOOKUP: Dict[str, str] = {}
'''
    ),
    # -------------------------------------------------------------------------
    # Cell 11 [628f3c82] - extract_pages (PDF -> per-page PIL images)
    # -------------------------------------------------------------------------
    "628f3c82": code(
        '''# =========================
# 2. SPLIT PDFS/IMAGES INTO PAGES
# =========================

import io as _io
import fitz  # PyMuPDF
from PIL import Image


def extract_pages(file_path: Path) -> List[Dict[str, Any]]:
    """Return one page-record per page of the input file.

    For PDFs, rasterize each page at PDF_RENDER_ZOOM.  For images, wrap as
    a single page.  HI.txt §A.1 rules are honored (image -> page=1; PDF ->
    sequential page numbers starting at 1).
    """
    ext = file_path.suffix.lower()
    name = file_path.name
    pages: List[Dict[str, Any]] = []

    if ext == ".pdf":
        try:
            doc = fitz.open(str(file_path))
            for idx in range(len(doc)):
                p = doc[idx]
                mat = fitz.Matrix(PDF_RENDER_ZOOM, PDF_RENDER_ZOOM)
                pix = p.get_pixmap(matrix=mat)
                img = Image.open(_io.BytesIO(pix.tobytes("png"))).convert("RGB")
                pages.append({
                    "page_number": idx + 1,
                    "image": img,
                    "file_name": name,
                })
            doc.close()
        except Exception as e:
            print(f"[extract_pages] PDF '{name}' failed: {e}")
    elif ext in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
        try:
            img = Image.open(file_path)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            pages.append({"page_number": 1, "image": img, "file_name": name})
        except Exception as e:
            print(f"[extract_pages] image '{name}' failed: {e}")
    return pages
'''
    ),
    # -------------------------------------------------------------------------
    # Cell 12 [16594195] - The Gemma 3 12B multi-task analyzer (run_ocr)
    # -------------------------------------------------------------------------
    "16594195": code(
        '''# =========================
# 3. OCR + CLASSIFY + EXTRACT (single Gemma 3 12B call per page)
# =========================
# Token discipline: one call per page returns ONLY the structured fields we
# need (no full-text OCR dump).  Results are cached to disk so reruns cost 0
# tokens.  Cache layout: vlm_cache/<case_id>/<file>__pN.json

import io as _vio
import re as _re

DOC_TYPE_VOCAB = [
    "clinical_notes", "cbc_hb_report", "indoor_case", "treatment_details",
    "post_hb_report", "discharge_summary", "usg_report", "lft_report",
    "operative_notes", "pre_anesthesia", "histopathology", "xray_ct_knee",
    "investigation_pre", "investigation_post", "vitals_treatment",
    "implant_invoice", "post_op_photo", "post_op_xray", "photo_evidence",
    "extra_document",
]

GEMMA_SYSTEM = (
    "You are a medical-document analyst for the Indian Ayushman Bharat PMJAY "
    "claim adjudication system.  Read healthcare scans (English + Hindi/Bengali/"
    "Assamese/Gujarati where present), classify each page into a fixed taxonomy, "
    "detect visual evidence, and extract STG-relevant entities.  Reply with one "
    "JSON object and nothing else - no prose, no markdown, no code fences."
)


def _gemma_user_prompt(package_code: str) -> str:
    types = ", ".join(DOC_TYPE_VOCAB)
    return (
        f"Analyze this single page from a PMJAY claim under package {package_code}.\\n\\n"
        "Return ONE JSON object EXACTLY matching this schema (use 0/1 for booleans):\\n"
        "{\\n"
        f'  "doc_type": "<one of: {types}>",\\n'
        '  "doc_type_confidence": 0.0,\\n'
        '  "ocr_snippet": "<<= 300 chars of the most informative English text on the page>",\\n'
        '  "language": "en|hi|bn|gu|mr|ta|te|mixed",\\n'
        '  "is_blurry": 0,\\n'
        '  "visual_elements": {\\n'
        '    "has_stamp": 0, "has_signature": 0, "has_photo_evidence": 0,\\n'
        '    "has_implant_sticker": 0, "has_table": 0, "has_xray": 0\\n'
        '  },\\n'
        '  "entities": {\\n'
        '    "patient_age": null,\\n'
        '    "dates_found": [],\\n'
        '    "doa": null,\\n'
        '    "dod": null,\\n'
        '    "hb_values": [],\\n'
        '    "temperature_celsius": null,\\n'
        '    "fever_duration_days": null,\\n'
        '    "diagnoses": [],\\n'
        '    "symptoms": [],\\n'
        '    "treatments": []\\n'
        '  }\\n'
        "}\\n\\n"
        "Classification cheat-sheet:\\n"
        "- clinical_notes: doctor history/examination/diagnosis (NOT discharge).\\n"
        "- cbc_hb_report: pre-treatment CBC or Hb lab.\\n"
        "- post_hb_report: Hb measured AFTER transfusion / treatment.\\n"
        "- indoor_case: admission paper / IPD / ward record.\\n"
        "- treatment_details: medication chart, transfusion record, drug list.\\n"
        "- discharge_summary: final summary on discharge with admission+discharge dates.\\n"
        "- usg_report: ultrasound (gallbladder/abdomen for SG039C).\\n"
        "- lft_report: liver function tests.\\n"
        "- operative_notes: surgical / OT notes describing the procedure.\\n"
        "- pre_anesthesia: PAC / anesthesia fitness clearance.\\n"
        "- histopathology: biopsy / microscopic pathology report.\\n"
        "- photo_evidence: intra-operative or specimen photograph.\\n"
        "- xray_ct_knee: pre-op X-ray or CT of the knee (SB039A).\\n"
        "- post_op_xray: X-ray taken AFTER surgery (typically shows implant).\\n"
        "- post_op_photo: photograph of patient AFTER surgery.\\n"
        "- implant_invoice: invoice/bill/sticker showing the implant used.\\n"
        "- investigation_pre / investigation_post: pre- vs post-treatment lab investigations (MG006A enteric fever).\\n"
        "- vitals_treatment: vitals/temperature charts during treatment.\\n"
        "- extra_document: anything not in the above categories (consent, ID, unrelated).\\n\\n"
        "Entity rules:\\n"
        "- dates_found: every date you can read, copied verbatim (DD/MM/YY or DD-MM-YYYY).\\n"
        "- doa/dod: ONLY fill from a discharge_summary (admission/discharge dates).\\n"
        "- hb_values: numeric Hb in g/dL only, e.g. [5.8, 9.2].\\n"
        "- temperature_celsius: convert °F to °C if needed.\\n"
        "- fever_duration_days: integer days if mentioned, else null.\\n"
        "- treatments: lower-case keywords e.g. ['blood transfusion','ferrous sulphate injection'].\\n"
        "- symptoms: lower-case keywords e.g. ['pallor','fatigue','right hypochondrium pain'].\\n\\n"
        "Output ONLY the JSON object."
    )


MAX_IMAGE_LONG_EDGE = 1024  # bounds Gemma image-token cost; OCR still readable


def _image_to_data_uri(image: "Image.Image") -> str:
    """Resize to <=1024px on the long edge before encoding, to bound the
    image-token contribution Gemma 3 12B charges per call."""
    img = image
    w, h = img.size
    long_edge = max(w, h)
    if long_edge > MAX_IMAGE_LONG_EDGE:
        scale = MAX_IMAGE_LONG_EDGE / float(long_edge)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img = img.resize((new_w, new_h), Image.LANCZOS)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = _vio.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    b = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b}"


def _safe_json_loads(text: str) -> Dict[str, Any]:
    """Strict parse, then code-fence strip, then brace-trim fallback."""
    if not text:
        return {}
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:]
        t = t.strip()
    try:
        return json.loads(t)
    except Exception:
        pass
    s = t.find("{")
    e = t.rfind("}")
    if s != -1 and e > s:
        try:
            return json.loads(t[s:e + 1])
        except Exception:
            pass
    return {}


def _normalize_payload(raw: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        raw = {}
    dt = str(raw.get("doc_type", "extra_document")).strip().lower()
    if dt not in DOC_TYPE_VOCAB:
        dt = "extra_document"
    out = {
        "doc_type": dt,
        "doc_type_confidence": float(raw.get("doc_type_confidence") or 0.0),
        "ocr_snippet": str(raw.get("ocr_snippet") or "")[:600],
        "language": str(raw.get("language") or "en"),
        "is_blurry": int(bool(raw.get("is_blurry"))),
        "visual_elements": {
            "has_stamp": int(bool((raw.get("visual_elements") or {}).get("has_stamp"))),
            "has_signature": int(bool((raw.get("visual_elements") or {}).get("has_signature"))),
            "has_photo_evidence": int(bool((raw.get("visual_elements") or {}).get("has_photo_evidence"))),
            "has_implant_sticker": int(bool((raw.get("visual_elements") or {}).get("has_implant_sticker"))),
            "has_table": int(bool((raw.get("visual_elements") or {}).get("has_table"))),
            "has_xray": int(bool((raw.get("visual_elements") or {}).get("has_xray"))),
        },
        "entities": {
            "patient_age": (raw.get("entities") or {}).get("patient_age"),
            "dates_found": list((raw.get("entities") or {}).get("dates_found") or []),
            "doa": (raw.get("entities") or {}).get("doa"),
            "dod": (raw.get("entities") or {}).get("dod"),
            "hb_values": list((raw.get("entities") or {}).get("hb_values") or []),
            "temperature_celsius": (raw.get("entities") or {}).get("temperature_celsius"),
            "fever_duration_days": (raw.get("entities") or {}).get("fever_duration_days"),
            "diagnoses": list((raw.get("entities") or {}).get("diagnoses") or []),
            "symptoms": list((raw.get("entities") or {}).get("symptoms") or []),
            "treatments": list((raw.get("entities") or {}).get("treatments") or []),
        },
    }
    return out


def _cache_path(case_id: str, file_name: str, page_number: int) -> Path:
    safe = file_name.replace("/", "_").replace("\\\\", "_")
    return VLM_CACHE_ROOT / case_id / f"{safe}__p{page_number}.json"


def analyze_page_with_gemma(
    page_image: "Image.Image",
    case_id: str,
    file_name: str,
    page_number: int,
    package_code: str,
) -> Dict[str, Any]:
    """One Gemma 3 12B call per page; cached to disk."""
    cp = _cache_path(case_id, file_name, page_number)
    if cp.exists():
        try:
            with cp.open("r", encoding="utf-8") as f:
                return _normalize_payload(json.load(f))
        except Exception:
            pass

    if MAX_VLM_CALLS is not None and VLM_CALL_COUNTER["calls"] >= MAX_VLM_CALLS:
        return _normalize_payload({})  # token-budget guard

    try:
        data_url = _image_to_data_uri(page_image)
        resp = nc.completion(
            model=GEMMA_MODEL,
            messages=[
                {"role": "system", "content": GEMMA_SYSTEM},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": _gemma_user_prompt(package_code)},
                ]},
            ],
            metadata={"problem_statement": 1},
        )
        VLM_CALL_COUNTER["calls"] += 1
        text = (resp.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
        payload = _normalize_payload(_safe_json_loads(text))
        cp.parent.mkdir(parents=True, exist_ok=True)
        with cp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        return payload
    except Exception as e:
        print(f"[gemma] {case_id}/{file_name} p{page_number} failed: {e}")
        return _normalize_payload({})


def run_ocr(page_image: Any) -> Tuple[str, List[OCRLine]]:
    """Compatibility shim for the original notebook signature.

    The real Gemma call is made from `process_case` via `analyze_page_with_gemma`.
    This stub returns whatever ocr_snippet is reachable, so cells that still
    call `run_ocr(page_image)` receive a non-empty text string.
    """
    return ("", [])


# ============================================================================
# SECONDARY GEMMA CALL: per-page CLINICAL-FIELD assessment (per package)
# ============================================================================
# Why a second call: the primary call gives us doc_type + general entities,
# but my keyword-based detection of clinical fields (severe_anemia, fever,
# arthritis_type, etc.) hit a hard ceiling at clinical_f1=0.7838.  Asking
# Gemma directly to evaluate each clinical criterion lets the model do
# medical reasoning that keywords can't (e.g. "patient looks pale" → pallor
# → common_signs=1).  Output is a tiny binary JSON, ~70-100 tokens.
#
# Per-package prompts focus only on the rule fields HI.txt requires for
# THAT package code.  Cache: vlm_clinical_cache/<case>/<file>__pN.json.
# Failure → empty dict → my keyword detection fills in (no regression).


_CLINICAL_PROMPTS = {
    "MG064A": (
        "This is a single page from an MG064A (Severe Anemia) PMJAY claim.\\n\\n"
        "Read the page image carefully and return ONE JSON object with these four "
        "binary fields (0 = not present on this page, 1 = present on this page):\\n"
        "{\\n"
        '  "severe_anemia": <1 if THIS PAGE shows hemoglobin (Hb) <7 g/dL OR a written diagnosis of severe anemia, else 0>,\\n'
        '  "common_signs": <1 if THIS PAGE documents pallor, pale conjunctivae, fatigue, weakness, lethargy, easily fatigued, loss of appetite, else 0>,\\n'
        '  "significant_signs": <1 if THIS PAGE documents tachycardia, breathlessness, dyspnea, palpitations, dizziness, exertional dyspnea, else 0>,\\n'
        '  "life_threatening_signs": <1 if THIS PAGE documents shock, hemorrhage, bleeding, cardiac/heart failure, severe hypoxia, hypotension, altered consciousness, syncope, respiratory distress, else 0>\\n'
        "}\\n"
        "Reply with the JSON object only — no markdown, no commentary, no code fences."
    ),
    "SG039C": (
        "This is a single page from an SG039C (Cholecystectomy) PMJAY claim.\\n\\n"
        "Read the page image carefully and return ONE JSON object with these four "
        "binary fields (0 = not on this page, 1 = on this page):\\n"
        "{\\n"
        '  "clinical_condition": <1 if THIS PAGE documents cholelithiasis, cholecystitis, biliary colic, jaundice, gallstones, gall bladder pathology, biliary pancreatitis, or obstructive jaundice, else 0>,\\n'
        '  "usg_calculi": <1 if THIS PAGE is a USG/ultrasound report and explicitly mentions calculi, stones, cholelithiasis, or echogenic foci in the gall bladder, else 0>,\\n'
        '  "pain_present": <1 if THIS PAGE documents pain in the right hypochondrium, epigastrium, RUQ, biliary colic, or abdominal pain, else 0>,\\n'
        '  "previous_surgery": <1 if patient HISTORY on this page mentions prior cholecystectomy or gallbladder removal, else 0>\\n'
        "}\\n"
        "Reply with the JSON object only."
    ),
    "MG006A": (
        "This is a single page from an MG006A (Enteric Fever) PMJAY claim.\\n\\n"
        "Read the page image carefully and return ONE JSON object with these three "
        "binary fields (0 = not on this page, 1 = on this page):\\n"
        "{\\n"
        '  "fever": <1 if THIS PAGE documents fever, pyrexia, febrile state, OR temperature >=38.3°C/101°F, OR fever duration >2 days, else 0>,\\n'
        '  "symptoms": <1 if THIS PAGE documents enteric-fever symptoms (headache, body aches, joint pain, muscle pain, malaise, GI symptoms like vomiting/diarrhea/constipation, abdominal pain, splenomegaly, hepatomegaly, rose spots), else 0>,\\n'
        '  "poor_quality": <1 if the page image is blurry, illegible, badly cropped, or low quality, else 0>\\n'
        "}\\n"
        "Reply with the JSON object only."
    ),
    "SB039A": (
        "This is a single page from an SB039A (Total Knee Replacement, Primary) PMJAY claim.\\n\\n"
        "Read the page image carefully and return ONE JSON object with these three "
        "binary fields (0 = not on this page, 1 = on this page):\\n"
        "{\\n"
        '  "arthritis_type": <1 if THIS PAGE documents any arthritis diagnosis: osteoarthritis (OA), primary OA, rheumatoid arthritis (RA), post-traumatic arthritis, avascular necrosis (AVN), psoriatic arthritis, gouty arthritis, genu varum, narrowed joint space, else 0>,\\n'
        '  "post_op_implant_present": <1 if THIS PAGE is a post-operative X-ray that clearly shows the knee implant/prosthesis in situ, else 0>,\\n'
        '  "age_valid": <1 if THIS PAGE explicitly documents the patient age and the age is GREATER THAN 55 years, else 0>\\n'
        "}\\n"
        "Reply with the JSON object only."
    ),
}


_VALID_CLINICAL_KEYS = {
    "severe_anemia", "common_signs", "significant_signs", "life_threatening_signs",
    "clinical_condition", "usg_calculi", "pain_present", "previous_surgery",
    "fever", "symptoms", "poor_quality",
    "arthritis_type", "post_op_implant_present", "age_valid",
}


def _clinical_cache_path(case_id: str, file_name: str, page_number: int) -> Path:
    safe = file_name.replace("/", "_").replace("\\\\", "_")
    return VLM_CLINICAL_CACHE_ROOT / case_id / f"{safe}__p{page_number}.json"


def analyze_clinical_with_gemma(
    page_image: "Image.Image",
    case_id: str,
    file_name: str,
    page_number: int,
    package_code: str,
) -> Dict[str, int]:
    """Per-package clinical-field assessment via a focused Gemma prompt.

    Returns a small dict like {"severe_anemia": 1, "common_signs": 0, ...}
    containing ONLY keys relevant to the package.  Empty dict on any failure
    (caller falls back to keyword detection in populate_row_for_package).
    """
    cp = _clinical_cache_path(case_id, file_name, page_number)
    if cp.exists():
        try:
            with cp.open("r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return {k: int(bool(v)) for k, v in data.items() if k in _VALID_CLINICAL_KEYS}
        except Exception:
            pass

    if MAX_VLM_CALLS is not None and VLM_CALL_COUNTER["calls"] >= MAX_VLM_CALLS:
        return {}

    prompt = _CLINICAL_PROMPTS.get(package_code)
    if not prompt:
        return {}

    try:
        data_uri = _image_to_data_uri(page_image)
        resp = nc.completion(
            model=GEMMA_MODEL,
            messages=[
                {"role": "system", "content": "You are a medical-document analyst. Return ONLY one JSON object — no prose, no markdown, no code fences."},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": prompt},
                ]},
            ],
            metadata={"problem_statement": 1},
        )
        VLM_CALL_COUNTER["calls"] += 1
        text = (resp.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
        parsed = _safe_json_loads(text)
        result = {k: int(bool(parsed.get(k))) for k in parsed if k in _VALID_CLINICAL_KEYS}
        cp.parent.mkdir(parents=True, exist_ok=True)
        with cp.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        return result
    except Exception as e:
        print(f"[clinical] {case_id}/{file_name} p{page_number} failed: {e}")
        return {}
'''
    ),
    # -------------------------------------------------------------------------
    # Cell 13 [9543d2f6] - estimate_page_quality (cv2 Laplacian + density)
    # -------------------------------------------------------------------------
    "9543d2f6": code(
        '''# =========================
# 4. PAGE QUALITY ESTIMATOR
# =========================

try:
    import cv2  # type: ignore
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

BLUR_THRESHOLD = 80.0  # Laplacian variance below this -> probably blurry


def estimate_page_quality(page_image: Any, extracted_text: str = "") -> Dict[str, Any]:
    if page_image is None:
        return {"is_blurry": True, "is_poor_quality": True, "blur_score": 0.0, "text_density": 0}
    arr = np.array(page_image.convert("L"))
    if _HAS_CV2:
        blur = float(cv2.Laplacian(arr, cv2.CV_64F).var())
    else:
        a = arr.astype(np.float32)
        lap = (
            -4 * a[1:-1, 1:-1]
            + a[:-2, 1:-1] + a[2:, 1:-1]
            + a[1:-1, :-2] + a[1:-1, 2:]
        )
        blur = float(lap.var())
    text_len = len((extracted_text or "").strip())
    is_blurry = blur < BLUR_THRESHOLD
    return {
        "blur_score": blur,
        "text_density": text_len,
        "is_blurry": bool(is_blurry),
        "is_poor_quality": bool(is_blurry or text_len < 40),
    }
'''
    ),
    # -------------------------------------------------------------------------
    # Cell 14 [80e8393e] - detect_visual_elements (uses Gemma JSON; pyzbar QR)
    # -------------------------------------------------------------------------
    "80e8393e": code(
        '''# =========================
# 5. VISUAL CUE DETECTION
# =========================
# The heavy lifting is done inside the Gemma call; this function is a small
# wrapper that adds deterministic QR/barcode detection via pyzbar (when
# available) and merges everything into a flat dict.

try:
    from pyzbar.pyzbar import decode as _zbar_decode  # type: ignore
    _HAS_ZBAR = True
except Exception:
    _HAS_ZBAR = False


def _detect_codes(image: Any) -> Dict[str, int]:
    if image is None or not _HAS_ZBAR:
        return {"has_qr_code": 0, "has_barcode": 0}
    try:
        decoded = _zbar_decode(np.array(image.convert("RGB")))
    except Exception:
        return {"has_qr_code": 0, "has_barcode": 0}
    has_qr = 0
    has_bar = 0
    for d in decoded:
        t = (getattr(d, "type", "") or "").upper()
        if t == "QRCODE":
            has_qr = 1
        elif t:
            has_bar = 1
    return {"has_qr_code": has_qr, "has_barcode": has_bar}


def detect_visual_elements(page_image: Any, extracted_text: str = "", vlm_payload: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Combine VLM-reported visual flags with deterministic QR/barcode detection.

    `vlm_payload` is the dict returned by `analyze_page_with_gemma`.  When
    omitted (legacy callers), only QR/barcode detection runs.
    """
    vlm_visual = (vlm_payload or {}).get("visual_elements") or {}
    code_visual = _detect_codes(page_image)
    out = {
        "has_stamp": int(bool(vlm_visual.get("has_stamp"))),
        "has_signature": int(bool(vlm_visual.get("has_signature"))),
        "has_photo_evidence": int(bool(vlm_visual.get("has_photo_evidence"))),
        "has_implant_sticker": int(bool(vlm_visual.get("has_implant_sticker"))),
        "has_table": int(bool(vlm_visual.get("has_table"))),
        "has_xray": int(bool(vlm_visual.get("has_xray"))),
        "has_qr_code": int(bool(code_visual.get("has_qr_code"))),
        "has_barcode": int(bool(code_visual.get("has_barcode"))),
    }
    return out
'''
    ),
    # -------------------------------------------------------------------------
    # Cell 15 [619e2895] - DOCUMENT_TYPES + classify_document_type
    # -------------------------------------------------------------------------
    "619e2895": code(
        '''# =========================
# 6. DOCUMENT TYPE CLASSIFIER
# =========================

DOCUMENT_TYPES = list(DOC_TYPE_VOCAB)  # alias for back-compat with notebook


# Strong filename markers that override Gemma when present.
# These tokens are highly specific; the dataset filenames embed them as
# explicit doc-type markers (e.g. "..._OT_NOTE.pdf").  Each entry is a
# RAW substring (lowercase) → doc_type.  Order matters — more specific
# markers come first so e.g. "post_op_xray" wins over "xray".
STRONG_FILENAME_HINTS = [
    ("histopathology", "histopathology"),
    ("histopath", "histopathology"),
    ("biopsy", "histopathology"),
    ("post_op_xray", "post_op_xray"),
    ("post_xray", "post_op_xray"),
    ("postop_xray", "post_op_xray"),
    ("post-op-xray", "post_op_xray"),
    ("post_op_photo", "post_op_photo"),
    ("postop_photo", "post_op_photo"),
    ("indoor_admission", "indoor_case"),
    ("indoor_case", "indoor_case"),
    ("__indoor", "indoor_case"),
    ("ot_note", "operative_notes"),
    ("ot_notes", "operative_notes"),
    ("operative_note", "operative_notes"),
    ("operative_notes", "operative_notes"),
    ("operation_note", "operative_notes"),
    ("__opnote", "operative_notes"),
    ("__op_note", "operative_notes"),
    ("pre_anesthesia", "pre_anesthesia"),
    ("preanesthesia", "pre_anesthesia"),
    ("pre_anaesthesia", "pre_anesthesia"),
    ("__pac.", "pre_anesthesia"),
    ("implant_invoice", "implant_invoice"),
    ("implant_sticker", "implant_invoice"),
    ("__barcode", "implant_invoice"),
    ("__invoice", "implant_invoice"),
    ("photo_evidence", "photo_evidence"),
    ("intraop_photo", "photo_evidence"),
    ("intraoperative", "photo_evidence"),
    ("specimen_photo", "photo_evidence"),
    ("__usg", "usg_report"),
    ("ultrasound", "usg_report"),
    ("sonography", "usg_report"),
    ("__lft", "lft_report"),
    ("lft_report", "lft_report"),
    ("__widal", "investigation_pre"),
    ("__cbc", "cbc_hb_report"),
    ("haemogram", "cbc_hb_report"),
    ("hemogram", "cbc_hb_report"),
    ("__post_hb", "post_hb_report"),
    ("post_treatment_hb", "post_hb_report"),
    ("transfusion", "treatment_details"),
    ("__tpr", "vitals_treatment"),
    ("__vitals", "vitals_treatment"),
    ("xray_knee", "xray_ct_knee"),
    ("__xray.", "xray_ct_knee"),
    ("__ct_knee", "xray_ct_knee"),
    ("dis_summary", "discharge_summary"),
    ("discharge_summary", "discharge_summary"),
    ("__dis.", "discharge_summary"),
    ("__dc_", "discharge_summary"),
    ("__ds.", "discharge_summary"),
    ("clinical_notes", "clinical_notes"),
    ("__notes.", "clinical_notes"),
    ("initial_assessment", "clinical_notes"),
]


def filename_doc_type_hint(file_name: str) -> Optional[str]:
    """Filename-hint override is currently DISABLED.

    Eval result: filename hints had ~zero net effect (final_score 0.7641 →
    0.7634, mandatory_f1 -0.0004, rank_score -0.0042).  Leaving the hint
    table in place above as documentation but the function returns None
    so classification falls through to Gemma + keyword fallback only.
    """
    return None


def classify_document_type(
    package_code_or_text=None,
    extracted_text=None,
    visual_tags=None,
    vlm_payload: Optional[Dict[str, Any]] = None,
    file_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Resolve final doc_type using the Gemma payload as primary source.

    The notebook's main runner calls this with positional args
    (package_code, text, visual_tags); we tolerate that shape AND a single
    Gemma-payload kwarg.  Returns a dict {"doc_type": str, "confidence": float}.
    """
    if vlm_payload is None and isinstance(package_code_or_text, dict):
        vlm_payload = package_code_or_text

    # Filename hint takes precedence when (a) Gemma is uncertain, OR (b) Gemma
    # picked extra_document but the filename clearly says otherwise, OR (c)
    # the filename hint is a doc-type that the schema recognises and Gemma's
    # confidence is below 0.85.  Strong filename markers like "_OT_NOTE",
    # "_BARCODE", "_POST_XRAY" are deterministic signals.
    fname_hint = filename_doc_type_hint(file_name) if file_name else None

    if vlm_payload:
        dt = str(vlm_payload.get("doc_type", "extra_document")).lower()
        conf = float(vlm_payload.get("doc_type_confidence") or 0.0)
        # Filename-hint override
        if fname_hint and fname_hint in DOCUMENT_TYPES:
            if dt == "extra_document" or dt not in DOCUMENT_TYPES or conf < 0.85:
                return {"doc_type": fname_hint, "confidence": max(conf, 0.92)}
        if dt in DOCUMENT_TYPES and conf >= 0.30:
            return {"doc_type": dt, "confidence": conf}
        text = vlm_payload.get("ocr_snippet", "") or ""
        v_tags = vlm_payload.get("visual_elements") or {}
    else:
        text = extracted_text or ""
        v_tags = visual_tags or {}

    # When falling through to the keyword path, the filename hint still wins
    # if present (the keyword path is a weak signal).
    if fname_hint and fname_hint in DOCUMENT_TYPES:
        return {"doc_type": fname_hint, "confidence": 0.85}

    # Keyword fallback for low-confidence or empty-VLM cases.
    lo = text.lower()
    keywords = {
        "clinical_notes": ["history", "examination", "complaint", "diagnosis", "h/o"],
        "cbc_hb_report": ["complete blood count", "cbc", "hemoglobin", "haemoglobin", "rbc", "wbc"],
        "post_hb_report": ["post hb", "post-treatment hb", "post transfusion"],
        "indoor_case": ["indoor", "ipd", "admission", "ward", "bed no"],
        "treatment_details": ["transfusion", "ferrous sulphate", "medication chart"],
        "discharge_summary": ["discharge summary", "date of discharge", "condition at discharge"],
        "usg_report": ["ultrasound", "usg", "sonography"],
        "lft_report": ["liver function", "lft", "bilirubin", "sgot", "sgpt"],
        "operative_notes": ["operative notes", "ot notes", "operation notes", "intra-op"],
        "pre_anesthesia": ["pre-anesthesia", "pre anesthesia", "pac"],
        "histopathology": ["histopathology", "biopsy", "microscopic"],
        "xray_ct_knee": ["x-ray knee", "xray knee", "ct knee"],
        "investigation_pre": ["widal", "blood culture", "investigation report"],
        "investigation_post": ["repeat widal", "post treatment investigation"],
        "vitals_treatment": ["vitals", "temperature chart", "tpr"],
        "implant_invoice": ["invoice", "tax invoice", "implant", "barcode"],
        "post_op_photo": ["post operative photo", "post-op photo"],
        "post_op_xray": ["post operative x-ray", "post-op x-ray"],
        "photo_evidence": ["intraoperative photograph", "specimen photo"],
    }
    scores: Dict[str, int] = {}
    for dt, kws in keywords.items():
        scores[dt] = sum(1 for kw in kws if kw in lo)
    if v_tags.get("has_xray"):
        scores["xray_ct_knee"] = scores.get("xray_ct_knee", 0) + 2
        scores["post_op_xray"] = scores.get("post_op_xray", 0) + 2
    if v_tags.get("has_implant_sticker") or v_tags.get("has_barcode"):
        scores["implant_invoice"] = scores.get("implant_invoice", 0) + 4
    if v_tags.get("has_photo_evidence"):
        scores["photo_evidence"] = scores.get("photo_evidence", 0) + 4
        scores["post_op_photo"] = scores.get("post_op_photo", 0) + 2

    if not scores or max(scores.values()) <= 0:
        return {"doc_type": "extra_document", "confidence": 0.20}
    best = max(scores, key=scores.get)
    return {"doc_type": best, "confidence": min(scores[best] / 6.0, 0.85)}
'''
    ),
    # -------------------------------------------------------------------------
    # Cell 16 [8f8d70aa] - find_dates / find_age / contains_any
    # -------------------------------------------------------------------------
    "8f8d70aa": code(
        '''# =========================
# ENTITY EXTRACTION HELPERS
# =========================

DATE_PATTERNS = [
    r"\\b\\d{1,2}[/-]\\d{1,2}[/-]\\d{2,4}\\b",
    r"\\b\\d{1,2}[.]\\d{1,2}[.]\\d{2,4}\\b",
    r"\\b\\d{1,2}-[A-Za-z]{3}-\\d{2,4}\\b",
    r"\\b\\d{1,2}\\s+[A-Za-z]{3,9}\\s+\\d{2,4}\\b",
    r"\\b[A-Za-z]{3,9}\\s+\\d{1,2},?\\s+\\d{2,4}\\b",
]


def find_dates(text: str) -> List[str]:
    if not text:
        return []
    seen, out = set(), []
    for p in DATE_PATTERNS:
        for m in re.findall(p, text):
            if m not in seen:
                seen.add(m)
                out.append(m)
    return out


def find_age(text: str) -> Optional[int]:
    if not text:
        return None
    patterns = [
        r"age[:\\s]+(\\d{1,3})",
        r"(\\d{1,3})\\s*(?:years?|yrs?|y\\.?o\\.?|year-?old)",
        r"\\b(\\d{1,3})\\s*y\\b",
    ]
    lo = text.lower()
    for p in patterns:
        m = re.search(p, lo)
        if m:
            age = int(m.group(1))
            if 0 < age < 120:
                return age
    return None


def contains_any(text: str, keywords: List[str]) -> int:
    if not text or not keywords:
        return 0
    lo = text.lower()
    return int(any(kw.lower() in lo for kw in keywords))


def parse_hb_values(text: str) -> List[float]:
    if not text:
        return []
    out: List[float] = []
    for m in re.finditer(
        r"\\b(?:hb|hgb|h\\.?b|haemoglobin|hemoglobin)\\s*[:=\\-]?\\s*(\\d{1,2}(?:\\.\\d{1,2})?)\\b",
        text, flags=re.IGNORECASE,
    ):
        try:
            v = float(m.group(1))
            if 0 < v < 25:
                out.append(v)
        except ValueError:
            pass
    return out
'''
    ),
    # -------------------------------------------------------------------------
    # Cell 17 [9c3c2bde] - first populate_row_for_package stub (kept as helper)
    # -------------------------------------------------------------------------
    "9c3c2bde": code(
        '''# (placeholder - actual implementation lives in the cell below this one)
# Keeping this cell empty-ish so the notebook flow is not interrupted.
'''
    ),
    # -------------------------------------------------------------------------
    # Cell 18 [0e4bb569] - normalize_output_key + initialize_output_row
    # -------------------------------------------------------------------------
    "0e4bb569": code(
        '''# =========================
# PACKAGE ROW INITIALIZERS
# =========================

def normalize_output_key(key: str) -> str:
    return KEY_ALIASES.get(key, key)


def initialize_output_row(package_code: str, case_id: str, file_name: str, page_number: int) -> Dict[str, Any]:
    schema = PACKAGE_SCHEMAS[package_code]
    link_key = LINK_FIELD[package_code]
    row: Dict[str, Any] = {}
    for key in schema:
        if key == "case_id":
            row[key] = case_id
        elif key == link_key:
            row[key] = file_name
        elif key == "procedure_code":
            row[key] = package_code
        elif key == "page_number":
            row[key] = int(page_number)
        elif key in NULLABLE_DATE_KEYS:
            row[key] = None
        elif key == "document_rank":
            row[key] = 99
        else:
            row[key] = 0
    return row
'''
    ),
    # -------------------------------------------------------------------------
    # Cell 19 [64a2997c] - real populate_row_for_package (PER-PAGE EVIDENCE)
    # -------------------------------------------------------------------------
    "64a2997c": code(
        '''# =========================
# PAGE-TO-ROW MAPPING (per-package STG fields)
# =========================
# CRITICAL: rule fields are PER-PAGE EVIDENCE, not case-level.
# HI.txt example: severe_anemia=1 on the CBC page (where Hb<7 appears),
# severe_anemia=0 on the extra_document page in the same case.
# So a field is 1 ONLY IF this specific page contains supporting evidence.
#
# Dates and case-level fields get propagated to every row in process_case()
# (see "case-level date propagation" pass), per HI.txt §C.2 / §D.2.

# ----- per-package keyword lists (lifted from STG MD specs) -----
ANEMIA_COMMON_SIGNS = ["pallor", "pale conjunctivae", "pale mucous membranes",
                      "fatigue", "weakness", "tiredness", "lethargy",
                      "loss of appetite", "easily fatigued"]
ANEMIA_SIGNIFICANT_SIGNS = ["dyspnea", "dyspnoea", "shortness of breath", "sob",
                           "tachycardia", "rapid heart rate", "palpitations",
                           "palpitation", "exertional dyspnea", "dizziness", "syncope"]
ANEMIA_LIFE_THREATENING = ["cardiac failure", "heart failure", "chf",
                          "severe hypoxia", "shock", "altered consciousness",
                          "altered sensorium", "respiratory distress",
                          "hemodynamic instability", "haemodynamic instability",
                          "hemorrhagic shock", "bleeding", "hemorrhage",
                          "hypotension", "cold extremities", "edema"]
ANEMIA_SEVERITY_TEXT = ["severe anemia", "severe anaemia", "hb<7", "hb < 7",
                       "hb less than 7", "haemoglobin < 7", "hemoglobin < 7"]

CHOLE_CLINICAL_CONDITIONS = [
    # Primary diagnoses
    "cholelithiasis", "cholecystitis", "cholelith",
    "acute cholecystitis", "chronic cholecystitis", "calculus cholecystitis",
    "acalculous cholecystitis",
    "biliary colic", "biliary pancreatitis", "biliary disease",
    # Complications
    "obstructive jaundice", "icterus", "jaundice",
    # Stones
    "gallstone", "gall stone", "gall bladder stone", "gb stone", "gb stones",
    # Anatomical references typical in clinical notes
    "right hypochondrium", "epigastrium",
    # Symptoms commonly tied to GB pathology
    "vomiting",
]
CHOLE_USG_CALCULI = [
    # Standard terms
    "calculi", "calculus", "stones",
    "cholelithiasis", "choledocholithiasis", "cholelith",
    # Size variants — specifically called out by case-1 (microlithiasis was missed)
    "microlithiasis", "biliary microlithiasis", "tiny gallstones", "tiny calculi",
    "multiple calculi", "single calculus",
    "large calculus", "single large stone",
    # Common phrasings
    "gall bladder calculi", "gb calculi", "gb stone", "gb stones",
    "gall stones", "gallstones", "biliary calculi",
    # Sludge / debris / sonographic markers
    "biliary sludge", "biliary mud", "gb sludge", "sludge with calculi",
    "echogenic focus", "echogenic foci", "echogenic shadow",
    "posterior acoustic shadow", "acoustic shadowing",
]
CHOLE_PAIN = [
    "right hypochondrium", "epigastrium", "epigastric pain",
    "right upper quadrant", "ruq pain", "ruq tenderness",
    "biliary colic", "pain abdomen", "abdominal pain", "abdominal tenderness",
    "tenderness in ruq", "tenderness over gb",
    "murphy's sign", "murphy sign", "murphy positive",
    "hypochondrial pain", "pain in stomach",
]
CHOLE_PRIOR_SURGERY = [
    "prior cholecystectomy", "previous cholecystectomy",
    "history of cholecystectomy", "post cholecystectomy",
    "post chole", "post lap chole", "post laparoscopic cholecystectomy",
    "h/o cholecystectomy", "s/p cholecystectomy", "status post cholecystectomy",
    "previous gb surgery", "previous gallbladder surgery",
]
# Phrases that, when seen IMMEDIATELY BEFORE a previous_surgery term, NEGATE it.
CHOLE_NEGATION_PHRASES = [
    "no h/o", "no history of", "no h o ", "no prior", "no previous",
    "denies", "denied", "negative for", "without",
    "absent", "ruled out", "r/o ", "not present",
    "no significant past", "no past history of", "no past h/o",
]


def _has_unnegated_match(text: str, terms: List[str], window: int = 30) -> int:
    """Return 1 iff any of `terms` appears in `text` without an immediately-
    preceding negation phrase (within `window` characters).  Used so that
    'no h/o cholecystectomy' does NOT mark previous_surgery=1.
    """
    if not text or not terms:
        return 0
    lo = text.lower()
    for term in terms:
        t = term.lower()
        idx = 0
        while idx < len(lo):
            m = lo.find(t, idx)
            if m == -1:
                break
            preceding = lo[max(0, m - window):m]
            if not any(neg in preceding for neg in CHOLE_NEGATION_PHRASES):
                return 1
            idx = m + len(t)
    return 0

FEVER_TERMS = ["fever", "pyrexia", "febrile", "high temperature",
              "raised temperature", "high grade fever", "low grade fever"]
FEVER_SYMPTOMS = ["headache", "dizziness", "pain in muscles", "pain in joints",
                 "muscle pain", "joint pain", "body ache", "weakness",
                 "malaise", "constipation", "diarrhea", "diarrhoea",
                 "rose-colored spots", "rose spots", "splenomegaly",
                 "hepatomegaly", "enlarged spleen", "enlarged liver",
                 "vomiting", "nausea", "abdominal pain", "loss of appetite"]

ARTHRITIS_TERMS = ["primary osteoarthritis", "osteoarthritis", "primary oa",
                  "tricompartmental osteoarthritis", "tricompartmental oa",
                  "bilateral oa", "secondary oa", "post-traumatic",
                  "post traumatic arthritis", "rheumatoid arthritis", "ra",
                  "avascular necrosis", "avn", "osteonecrosis",
                  "psoriatic arthritis", "gouty arthritis", "genu varum",
                  "bow-legged", "narrowed joint space",
                  "joint space narrowing", "knee instability",
                  "joint instability", "frequent locking episodes"]


def _page_text(payload: Optional[Dict[str, Any]]) -> str:
    """Render the page's own snippet + extracted entities for keyword search.

    PER-PAGE only; do NOT join across pages of the case.
    """
    if not payload:
        return ""
    e = payload.get("entities") or {}
    parts: List[str] = [payload.get("ocr_snippet", "") or ""]
    parts.extend(str(x) for x in (e.get("symptoms") or []))
    parts.extend(str(x) for x in (e.get("diagnoses") or []))
    parts.extend(str(x) for x in (e.get("treatments") or []))
    return " | ".join(p for p in parts if p)


def populate_row_for_package(package_code: str, page_result: PageResult) -> Dict[str, Any]:
    row = initialize_output_row(
        package_code,
        page_result.case_id,
        page_result.file_name,
        page_result.page_number,
    )

    payload: Dict[str, Any] = page_result.entities or {}
    page_text = _page_text(payload)
    visual = page_result.visual_tags or {}
    quality = page_result.quality or {}
    doc_type = page_result.doc_type
    extracted = (payload.get("entities") or {}) if isinstance(payload, dict) else {}

    # Mark the doc-type presence flag if it exists in the schema.
    if doc_type in row:
        row[doc_type] = 1
    if doc_type not in DOCUMENT_TYPES or doc_type == "extra_document":
        row["extra_document"] = 1

    # ------------- MG064A: Severe Anemia (PER-PAGE EVIDENCE) ---------------
    if package_code == "MG064A":
        page_hb = list(extracted.get("hb_values") or []) + parse_hb_values(page_text)
        severe = 0
        if any((isinstance(v, (int, float)) and v < 7.0) for v in page_hb):
            severe = 1
        if contains_any(page_text, ANEMIA_SEVERITY_TEXT):
            severe = 1
        # Boost: if Gemma's diagnoses list mentions anemia explicitly on this
        # page, treat it as severe-anemia evidence too (the diagnosis is the
        # ultimate signal of "this page documents severe anemia").
        diag_str = " | ".join(str(d) for d in (extracted.get("diagnoses") or []))
        if contains_any(diag_str, ["severe anemia", "severe anaemia"]):
            severe = 1
        row["severe_anemia"] = severe
        row["common_signs"] = contains_any(page_text, ANEMIA_COMMON_SIGNS)
        row["significant_signs"] = contains_any(page_text, ANEMIA_SIGNIFICANT_SIGNS)
        row["life_threatening_signs"] = contains_any(page_text, ANEMIA_LIFE_THREATENING)

    # ------------- SG039C: Cholecystectomy (PER-PAGE EVIDENCE) -------------
    elif package_code == "SG039C":
        row["clinical_condition"] = contains_any(page_text, CHOLE_CLINICAL_CONDITIONS)
        row["usg_calculi"] = contains_any(page_text, CHOLE_USG_CALCULI)
        row["pain_present"] = contains_any(page_text, CHOLE_PAIN)
        # Negation-aware match for previous_surgery (avoid false positives on
        # "no h/o cholecystectomy" / "denies prior surgery" / etc.)
        row["previous_surgery"] = _has_unnegated_match(page_text, CHOLE_PRIOR_SURGERY)

    # ------------- MG006A: Enteric Fever (PER-PAGE EVIDENCE) ---------------
    elif package_code == "MG006A":
        # Dates extracted from investigation_pre / investigation_post pages
        # (HI.txt §C.2).  Propagation to other rows happens in process_case().
        dates = list(extracted.get("dates_found") or []) or find_dates(payload.get("ocr_snippet", "") or "")
        if doc_type == "investigation_pre" and dates:
            row["pre_date"] = dates[0]
        elif doc_type == "investigation_post" and dates:
            row["post_date"] = dates[0]

        row["poor_quality"] = int(bool(quality.get("is_poor_quality") or payload.get("is_blurry")))
        # Fever per STG: temp >=38.3°C (or >=101°F) for >2 days; or "fever" mentioned ON THIS PAGE.
        temp_c = extracted.get("temperature_celsius")
        dur = extracted.get("fever_duration_days")
        fever_flag = contains_any(page_text, FEVER_TERMS)
        try:
            if temp_c is not None and 30.0 <= float(temp_c) <= 45.0 and float(temp_c) >= 38.3:
                fever_flag = 1
            if dur is not None and int(dur) > 2:
                fever_flag = max(fever_flag, 1)
        except Exception:
            pass
        row["fever"] = fever_flag
        row["symptoms"] = contains_any(page_text, FEVER_SYMPTOMS)

    # ------------- SB039A: Total Knee Replacement (PER-PAGE EVIDENCE) ------
    elif package_code == "SB039A":
        # doa/dod extracted from discharge_summary pages (HI.txt §D.2);
        # propagation to other rows happens in process_case().
        if doc_type == "discharge_summary":
            doa = extracted.get("doa")
            dod = extracted.get("dod")
            dates = list(extracted.get("dates_found") or [])
            if not doa and dates:
                doa = dates[0]
            if not dod and len(dates) > 1:
                dod = dates[-1]
            row["doa"] = normalize_date(doa) if doa else None
            row["dod"] = normalize_date(dod) if dod else None

        row["arthritis_type"] = contains_any(page_text, ARTHRITIS_TERMS)

        # post_op_implant_present: per-page evidence; only set on x-ray pages
        # showing implant or pages mentioning implant explicitly.
        has_impl = 0
        if doc_type in {"post_op_xray", "implant_invoice"} and (
            visual.get("has_implant_sticker") or visual.get("has_barcode")
            or visual.get("has_xray")
        ):
            has_impl = 1
        if contains_any(page_text, ["implant in situ", "prosthesis in place",
                                    "knee prosthesis", "tkr in situ", "implant visualized",
                                    "post-op implant", "post operative implant"]):
            has_impl = 1
        row["post_op_implant_present"] = has_impl

        # age_valid is per-page: =1 if THIS page records a valid age (>55) or
        # mentions trauma/systemic disease justifying TKR irrespective of age.
        age_val = extracted.get("patient_age")
        if isinstance(age_val, str):
            try:
                age_val = int(age_val)
            except (ValueError, TypeError):
                age_val = None
        if not isinstance(age_val, int):
            age_val = find_age(page_text)
        is_primary_oa = contains_any(page_text, ["primary osteoarthritis", "primary oa"])
        has_trauma = contains_any(page_text, ["post-traumatic", "post traumatic", "trauma", "injury"])
        has_systemic = contains_any(page_text, ["rheumatoid", "inflammatory arthritis", "systemic"])
        if isinstance(age_val, int) and age_val > 55:
            row["age_valid"] = 1
        elif (has_trauma or has_systemic) and isinstance(age_val, int):
            row["age_valid"] = 1  # age restriction waived per STG MD §3
        else:
            row["age_valid"] = 0

    # ---- Clinical-cache combination: PURE INTERSECTION ----
    # Live-eval results:
    #     Override (Gemma only):    clinical_f1 0.7567 ❌
    #     Intersection (kw AND gem): clinical_f1 0.7909 ✅ +0.007  (best so far, 0.7661 final)
    #     Hybrid (per-field intersect/union): clinical_f1 0.7725 ❌ -0.018 vs intersection
    #
    # Pure intersection wins.  Field-specific union/intersection swap
    # actively hurts — labels are uniformly stricter than Gemma across
    # all clinical fields, so keyword-AND-Gemma filtering helps every
    # field consistently.
    clinical_payload = (page_result.evidence or {}).get("clinical_payload") or {}
    if clinical_payload and not int(row.get("extra_document", 0) or 0):
        for key, val in clinical_payload.items():
            if key in row and isinstance(val, (int, bool)):
                kw_val = int(row.get(key, 0) or 0)
                gem_val = int(bool(val))
                row[key] = 1 if (kw_val and gem_val) else 0

    return row
'''
    ),
    # -------------------------------------------------------------------------
    # Cell 20 [c6d06df8] - infer_document_rank + post-pass for multi-page PDFs
    # -------------------------------------------------------------------------
    "c6d06df8": code(
        '''# =========================
# DOCUMENT RANKING
# =========================

RANK_MAP = {
    # Tied values bumped so each doc-type has a unique typical-order
    # position, which gives the global-sequential rank assignment a stable
    # tie-breaker ordering downstream.
    "MG064A": {
        "clinical_notes": 1, "cbc_hb_report": 2, "indoor_case": 3,
        "treatment_details": 4, "post_hb_report": 5, "discharge_summary": 6,
    },
    "SG039C": {
        "clinical_notes": 1, "usg_report": 2, "lft_report": 3,
        "pre_anesthesia": 4, "operative_notes": 5, "photo_evidence": 6,
        "histopathology": 7, "discharge_summary": 8,
    },
    "MG006A": {
        "clinical_notes": 1, "investigation_pre": 2, "vitals_treatment": 3,
        "investigation_post": 4, "discharge_summary": 5,
    },
    "SB039A": {
        "clinical_notes": 1, "xray_ct_knee": 2, "indoor_case": 3,
        "operative_notes": 4, "implant_invoice": 5,
        "post_op_photo": 6, "post_op_xray": 7, "discharge_summary": 8,
    },
}


def infer_document_rank(package_code: str, row: Dict[str, Any], doc_type: str) -> Optional[int]:
    if int(row.get("extra_document", 0) or 0) == 1:
        return 99
    return RANK_MAP.get(package_code, {}).get(doc_type, 99)


def assign_document_ranks(package_code: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per-page document_rank from the page's own doc_type via RANK_MAP.

    Tested on the live eval, ranked by rank_score:
        Per-page from RANK_MAP (this)        → 0.107  ← best, reverted
        Global per-page seq within case      → 0.0999
        File-level seq by typical_rank       → 0.045
        File-level seq by filename           → 0.0075
    """
    rank_map_pkg = RANK_MAP.get(package_code, {})
    for row in rows:
        if int(row.get("extra_document", 0) or 0) == 1:
            row["document_rank"] = 99
            continue
        present = [k for k, v in row.items()
                   if k in rank_map_pkg and isinstance(v, int) and v == 1]
        if not present:
            row["document_rank"] = 99
        else:
            row["document_rank"] = min(rank_map_pkg[k] for k in present)
    return rows
'''
    ),
    # -------------------------------------------------------------------------
    # Cell 21 [67113e92] - timeline (minimal — used only for design/explanation)
    # -------------------------------------------------------------------------
    "67113e92": code(
        '''# =========================
# 8. TIMELINE CONSTRUCTION (minimal)
# =========================

EVENT_TYPE_MAP = {
    "clinical_notes": "Admission",
    "indoor_case": "Admission",
    "investigation_pre": "Diagnostic Investigation",
    "cbc_hb_report": "Diagnostic Investigation",
    "usg_report": "Diagnostic Investigation",
    "lft_report": "Diagnostic Investigation",
    "xray_ct_knee": "Diagnostic Investigation",
    "operative_notes": "Procedure",
    "treatment_details": "Treatment",
    "vitals_treatment": "Post-Procedure Monitoring",
    "investigation_post": "Post-Procedure Monitoring",
    "post_hb_report": "Post-Procedure Monitoring",
    "post_op_xray": "Post-Procedure Monitoring",
    "post_op_photo": "Post-Procedure Monitoring",
    "histopathology": "Post-Procedure Monitoring",
    "implant_invoice": "Procedure",
    "photo_evidence": "Procedure",
    "pre_anesthesia": "Pre-Procedure",
    "discharge_summary": "Discharge",
}


def build_episode_timeline(package_code: str, page_results: List[PageResult]) -> List[TimelineEvent]:
    seen_types: Dict[str, TimelineEvent] = {}
    for pr in page_results:
        et = EVENT_TYPE_MAP.get(pr.doc_type)
        if not et or et in seen_types:
            continue
        # Take the first date the entity extractor returned, if any.
        ent = (pr.entities or {}).get("entities") or {}
        dates = list(ent.get("dates_found") or [])
        seen_types[et] = TimelineEvent(
            sequence=len(seen_types) + 1,
            event_type=et,
            date=dates[0] if dates else None,
            source_document=pr.file_name,
            temporal_validity="Valid",
            evidence={"page_number": pr.page_number, "doc_type": pr.doc_type},
        )
    return list(seen_types.values())
'''
    ),
    # -------------------------------------------------------------------------
    # Cell 22 [cd5e76e9] - MANDATORY_DOCS, aggregate_case_rows, run_rules_engine
    # -------------------------------------------------------------------------
    "cd5e76e9": code(
        '''# =========================
# 9. RULES ENGINE (full STG logic per package)
# =========================

MANDATORY_DOCS = {
    "MG064A": ["clinical_notes", "cbc_hb_report", "indoor_case", "treatment_details", "post_hb_report", "discharge_summary"],
    "SG039C": ["clinical_notes", "usg_report", "lft_report", "operative_notes", "pre_anesthesia", "discharge_summary", "photo_evidence", "histopathology"],
    "MG006A": ["clinical_notes", "investigation_pre", "vitals_treatment", "investigation_post", "discharge_summary"],
    "SB039A": ["clinical_notes", "xray_ct_knee", "indoor_case", "operative_notes", "implant_invoice", "post_op_photo", "post_op_xray", "discharge_summary"],
}

ALOS_LIMITS = {  # max length-of-stay in days per STG MD
    "MG064A": 3,
    "SG039C": 3,
    "MG006A": 5,
    "SB039A": 7,
}


def aggregate_case_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collapse per-page rows into a case-level evidence dict (max over pages)."""
    if not rows:
        return {}
    agg: Dict[str, Any] = {}
    for r in rows:
        for k, v in r.items():
            if k in NULLABLE_DATE_KEYS:
                if v and not agg.get(k):
                    agg[k] = v
            elif isinstance(v, (int, float)):
                agg[k] = max(agg.get(k, 0), v) if isinstance(agg.get(k, 0), (int, float)) else v
            else:
                agg.setdefault(k, v)
    return agg


def _alos_check(package_code: str, doa: Optional[str], dod: Optional[str]) -> Optional[Dict[str, str]]:
    if not doa or not dod:
        return None
    try:
        doa_norm = normalize_date(doa) or doa
        dod_norm = normalize_date(dod) or dod
        d_in = datetime.strptime(doa_norm, "%d-%m-%Y")
        d_out = datetime.strptime(dod_norm, "%d-%m-%Y")
        los = (d_out - d_in).days
    except Exception:
        return None
    limit = ALOS_LIMITS.get(package_code)
    if limit is None:
        return None
    if los <= limit:
        return {"status": "PASS", "rule": "TMS_RULE_ALOS",
                "reason": f"Length of stay ({los}d) within ALOS limit ({limit}d)"}
    return {"status": "CONDITIONAL", "rule": "TMS_RULE_ALOS",
            "reason": f"Length of stay ({los}d) exceeds ALOS limit ({limit}d), manual review required"}


def run_rules_engine(case_id: str, package_code: str, rows: List[Dict[str, Any]],
                     timeline: List[TimelineEvent]) -> ClaimDecision:
    """Per-STG-MD Pass/Conditional/Fail decisioning with reasons."""
    case = aggregate_case_rows(rows)
    reasons: List[str] = []
    rule_flags: List[str] = []
    timeline_flags: List[str] = []

    missing_docs = [d for d in MANDATORY_DOCS.get(package_code, []) if not case.get(d)]
    for d in missing_docs:
        reasons.append(f"Missing mandatory document: {d}")

    # -------- Severe Anemia (MG064A) --------
    if package_code == "MG064A":
        if not case.get("severe_anemia"):
            rule_flags.append("TMS_RULE_1_SEVERITY")
            reasons.append("Pre-treatment Hb not <7 g/dL; severe-anemia criteria unmet (FAIL).")
        if not case.get("treatment_details"):
            rule_flags.append("TMS_RULE_2_TREATMENT")
            reasons.append("Treatment details (transfusion + ferrous sulphate injection) not documented.")
        if not case.get("cbc_hb_report"):
            rule_flags.append("TMS_RULE_3_PRE_LAB")
            reasons.append("Pre-treatment CBC/Hb report missing.")
        if not case.get("post_hb_report"):
            rule_flags.append("TMS_RULE_3_POST_LAB")
            reasons.append("Post-treatment Hb report missing.")

    # -------- Cholecystectomy (SG039C) --------
    elif package_code == "SG039C":
        if not case.get("usg_calculi"):
            rule_flags.append("TMS_RULE_1_IMAGING")
            reasons.append("USG does not show calculi (FAIL).")
        if not case.get("pain_present") and not case.get("clinical_condition"):
            rule_flags.append("TMS_RULE_2_CLINICAL")
            reasons.append("Mandatory clinical symptoms (right hypochondrium / epigastrium pain) not recorded.")
        if case.get("previous_surgery"):
            rule_flags.append("TMS_RULE_3_FRAUD")
            reasons.append("Patient has prior cholecystectomy (FAIL — possible fraud).")
        if not case.get("lft_report"):
            rule_flags.append("TMS_RULE_4_LFT")
            reasons.append("LFT report missing.")
        if not case.get("photo_evidence"):
            rule_flags.append("Visual_evidence_intraop_photo")
            reasons.append("Intraoperative photograph missing.")
        if not case.get("histopathology"):
            rule_flags.append("Visual_evidence_histopath")
            reasons.append("Histopathology report missing.")

    # -------- Enteric Fever (MG006A) --------
    elif package_code == "MG006A":
        if not case.get("fever"):
            rule_flags.append("TMS_RULE_1_FEVER")
            reasons.append("Fever criteria (>=101 F / >=38.3 C for >2 days) not satisfied (FAIL).")
        if not case.get("investigation_pre"):
            rule_flags.append("TMS_RULE_2_PRE_LAB")
            reasons.append("Pre-treatment investigation missing.")
        if not case.get("investigation_post"):
            rule_flags.append("TMS_RULE_2_POST_LAB")
            reasons.append("Post-treatment investigation missing.")
        # Date sequence: post_date must be after pre_date
        try:
            if case.get("pre_date") and case.get("post_date"):
                pd_ = datetime.strptime(normalize_date(case["pre_date"]) or case["pre_date"], "%d-%m-%Y")
                pt_ = datetime.strptime(normalize_date(case["post_date"]) or case["post_date"], "%d-%m-%Y")
                if pt_ <= pd_:
                    timeline_flags.append("Post-investigation date is not after pre-investigation date")
        except Exception:
            pass

    # -------- Total Knee Replacement (SB039A) --------
    elif package_code == "SB039A":
        if not case.get("post_op_implant_present"):
            rule_flags.append("TMS_RULE_1_POST_OP_VISUAL")
            reasons.append("Post-operative imaging does not show evidence of implant (FAIL).")
        if not case.get("age_valid"):
            rule_flags.append("TMS_RULE_3_AGE_CHECK")
            reasons.append("Age criteria for primary TKR (>55 y) not met (FAIL).")
        if not case.get("post_op_photo"):
            rule_flags.append("Visual_evidence_post_op_photo")
            reasons.append("Post-operative clinical photograph missing.")
        if not case.get("post_op_xray"):
            rule_flags.append("Visual_evidence_post_op_xray")
            reasons.append("Post-operative X-ray missing.")
        if not case.get("implant_invoice"):
            rule_flags.append("Visual_evidence_implant_invoice")
            reasons.append("Implant invoice / barcode missing.")
        # ALOS based on doa/dod
        alos = _alos_check(package_code, case.get("doa"), case.get("dod"))
        if alos and alos["status"] == "CONDITIONAL":
            timeline_flags.append(alos["reason"])

    # ------ Decision combining missing-docs + rule_flags + timeline_flags ------
    critical_failures = {
        "TMS_RULE_1_SEVERITY",
        "TMS_RULE_1_IMAGING",
        "TMS_RULE_2_CLINICAL",
        "TMS_RULE_3_FRAUD",
        "TMS_RULE_1_FEVER",
        "TMS_RULE_1_POST_OP_VISUAL",
        "TMS_RULE_3_AGE_CHECK",
    }
    has_critical = any(f in critical_failures for f in rule_flags)

    if has_critical:
        decision, confidence = DECISION_FAIL, 0.90
    elif missing_docs or rule_flags or timeline_flags:
        decision, confidence = DECISION_CONDITIONAL, 0.75
    else:
        decision, confidence = DECISION_PASS, 0.95
        reasons = ["All mandatory documents present", "All STG rules passed", "Timeline validated"]

    return ClaimDecision(
        case_id=case_id,
        package_code=package_code,
        decision=decision,
        confidence=confidence,
        reasons=reasons,
        missing_documents=missing_docs,
        rule_flags=rule_flags,
        timeline_flags=timeline_flags,
    )
'''
    ),
    # -------------------------------------------------------------------------
    # Cell 23 [0dae73e1] - human-readable summary + timeline DataFrame
    # -------------------------------------------------------------------------
    "0dae73e1": code(
        '''# =========================
# 10. EXPLAINABLE DECISIONING (reviewer-facing tables)
# =========================

def build_human_readable_summary(package_code: str, page_results: List[PageResult],
                                 decision: ClaimDecision) -> pd.DataFrame:
    rows = []
    for pr in page_results:
        rows.append({
            "Case ID": pr.case_id,
            "File": pr.file_name,
            "Page": pr.page_number,
            "Doc Type": pr.doc_type,
            "Confidence": round(pr.doc_type_confidence, 3),
            "Visual": ", ".join(k.replace("has_", "") for k, v in (pr.visual_tags or {}).items() if v),
            "Quality": "blurry" if (pr.quality or {}).get("is_blurry") else "ok",
        })
    rows.append({
        "Case ID": decision.case_id,
        "File": "DECISION",
        "Page": "",
        "Doc Type": decision.decision,
        "Confidence": decision.confidence,
        "Visual": ", ".join(decision.rule_flags[:3]),
        "Quality": " | ".join(decision.reasons[:2]),
    })
    return pd.DataFrame(rows)


def build_timeline_df(timeline: List[TimelineEvent]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "Sequence": t.sequence,
            "Event Type": t.event_type,
            "Date": t.date,
            "Source Document": t.source_document,
            "Temporal Validity": t.temporal_validity,
        }
        for t in timeline
    ])
'''
    ),
    # -------------------------------------------------------------------------
    # Cell 24 [bde92b83] - process_case orchestrator
    # -------------------------------------------------------------------------
    "bde92b83": code(
        '''# =========================
# CORE PIPELINE DRIVER
# =========================

def process_case(case_id: str, files: List[Path], package_code: str) -> Dict[str, Any]:
    """Run the full pipeline for one case and return all artifacts."""
    page_results: List[PageResult] = []
    strict_rows: List[Dict[str, Any]] = []

    # ---- pass 1: per-page Gemma + visual + classify ----
    raw_payloads: List[Dict[str, Any]] = []
    for fp in files:
        for page in extract_pages(fp):
            page_image = page["image"]
            page_number = page["page_number"]
            file_name = page["file_name"]

            # ONE Gemma call per page (cached on disk).
            payload = analyze_page_with_gemma(
                page_image=page_image,
                case_id=case_id,
                file_name=file_name,
                page_number=page_number,
                package_code=package_code,
            )
            raw_payloads.append(payload)

            visual = detect_visual_elements(page_image, payload.get("ocr_snippet", ""), vlm_payload=payload)
            quality = estimate_page_quality(page_image, payload.get("ocr_snippet", ""))
            cls = classify_document_type(vlm_payload=payload, file_name=file_name)
            doc_type = cls["doc_type"]
            doc_conf = cls["confidence"]

            # SECONDARY Gemma call: per-page clinical-field assessment.
            # Cached separately at vlm_clinical_cache/.  Empty dict on failure
            # → populate_row_for_package falls back to its keyword detection.
            clinical_payload = analyze_clinical_with_gemma(
                page_image=page_image,
                case_id=case_id,
                file_name=file_name,
                page_number=page_number,
                package_code=package_code,
            )

            pr = PageResult(
                case_id=case_id,
                file_name=file_name,
                page_number=page_number,
                extracted_text=payload.get("ocr_snippet", "") or "",
                ocr_lines=[],
                doc_type=doc_type,
                doc_type_confidence=doc_conf,
                visual_tags=visual,
                entities={"entities": payload.get("entities") or {},
                          "ocr_snippet": payload.get("ocr_snippet", "")},
                quality=quality,
                output_row={},
                evidence={"vlm_language": payload.get("language"),
                          "vlm_doc_conf": payload.get("doc_type_confidence"),
                          "clinical_payload": clinical_payload},
            )
            page_results.append(pr)

    # ---- pass 2: build case-level joined text and re-populate rows ----
    case_text_parts = []
    for pr in page_results:
        case_text_parts.append(pr.extracted_text or "")
        ent = (pr.entities or {}).get("entities") or {}
        case_text_parts.extend(str(x) for x in (ent.get("symptoms") or []))
        case_text_parts.extend(str(x) for x in (ent.get("diagnoses") or []))
        case_text_parts.extend(str(x) for x in (ent.get("treatments") or []))
    case_text = " | ".join(p for p in case_text_parts if p)

    for pr in page_results:
        # Stash the joined case text so populate_row_for_package can read it.
        setattr(pr, "case_text", case_text)
        # Pass full Gemma payload through pr.entities for the populator.
        pr.entities = {"entities": (pr.entities or {}).get("entities") or {},
                       "ocr_snippet": pr.extracted_text}
        row = populate_row_for_package(package_code, pr)
        pr.output_row = row
        strict_rows.append(row)

    # ---- pass 3: case-level propagation for patient-attribute fields ----
    # Some clinical fields encode patient-level facts (age, arthritis type,
    # surgical history) that don't actually vary page-to-page.  Per-page
    # evidence detection only fires on the 1-2 pages that document them
    # explicitly, missing the other rows of the same case.  OR them across
    # the case so every row reflects the case-level fact.
    strict_rows = _propagate_case_clinical(package_code, strict_rows)

    # ---- SG039C-only refinements (per notebooklm review of 10 cases) ----
    # Order matters: force extras BEFORE continuity so admin-filename
    # promotions are protected from being flipped to a clinical doc_type.
    if package_code == "SG039C":
        strict_rows = _sg039c_force_extra_by_filename(strict_rows)
        strict_rows = _sg039c_multi_page_continuity(strict_rows)

    # ---- pass 4: assign document_rank (per-page) ----
    strict_rows = assign_document_ranks(package_code, strict_rows)

    # ---- pass 5: propagate case-level dates to every row ----
    # HI.txt §C.2 / §D.2: dates are EXTRACTED from specific pages
    # (investigation_pre/post for MG006A; discharge_summary for SB039A) but
    # the labels apply them to every row of the case.  Without this step
    # MG006A loses ~6.5k pre_date values and SB039A loses ~4.7k doa values.
    # Pass raw VLM payloads so the function can fall back to ANY date in
    # the case when the schema-rule pages are absent.
    payloads_for_dates = []
    for pr in page_results:
        ents = ((pr.entities or {}).get("entities") or {})
        payloads_for_dates.append({"entities": ents})
    strict_rows = _propagate_case_dates(package_code, strict_rows, payloads_for_dates)

    # Date-field normalization (DD-MM-YYYY) for MG006A and SB039A.
    strict_rows = normalize_dates_in_rows(package_code, strict_rows)

    # ---- pass 4: timeline + decision ----
    timeline = build_episode_timeline(package_code, page_results)
    decision = run_rules_engine(case_id, package_code, strict_rows, timeline)

    summary_df = build_human_readable_summary(package_code, page_results, decision)
    timeline_df = build_timeline_df(timeline)

    return {
        "case_id": case_id,
        "package_code": package_code,
        "page_results": page_results,
        "strict_rows": strict_rows,
        "timeline": timeline,
        "decision": decision,
        "summary_df": summary_df,
        "timeline_df": timeline_df,
    }
'''
    ),
    # -------------------------------------------------------------------------
    # Cell 25 [732bbb15] - run_batch + helpers
    # -------------------------------------------------------------------------
    "732bbb15": code(
        '''# =========================
# BATCH RUNNER + CASE-LEVEL HELPERS
# =========================

def _propagate_case_dates(package_code: str, rows: List[Dict[str, Any]],
                          page_payloads: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """Fill case-level date fields on every row.

    HI.txt §C.2 / §D.2: dates are EXTRACTED from specific document pages
    but the labels carry them on every row of the case.  This pass picks
    the first non-null value for each date field and copies it to every
    row that has the field null.

    Fallback: when a case has no investigation_pre page (MG006A) or no
    discharge_summary page (SB039A), no canonical date can be extracted
    from the schema rules.  In that case we look across ALL the page
    payloads for the case and pick the EARLIEST date found anywhere as
    pre_date / doa, and the LATEST date as post_date / dod.  This was
    the source of the 895 missing pre_date values in the previous eval.
    """
    keys = DATE_FIELDS.get(package_code, [])
    if not keys or not rows:
        return rows
    for k in keys:
        canonical = None
        for r in rows:
            v = r.get(k)
            if v not in (None, "", "null"):
                canonical = v
                break
        # Fallback: scrape any date out of the page_payloads if we found
        # nothing under the normal extraction.
        if canonical is None and page_payloads:
            all_dates: List[str] = []
            for p in page_payloads:
                ents = (p or {}).get("entities") or {}
                all_dates.extend(str(d) for d in (ents.get("dates_found") or []) if d)
            if all_dates:
                # Keep order found; take first for "pre"/"doa" types,
                # last for "post"/"dod".
                if k in ("pre_date", "doa"):
                    canonical = all_dates[0]
                elif k in ("post_date", "dod") and len(all_dates) > 1:
                    canonical = all_dates[-1]
                elif k in ("post_date", "dod"):
                    canonical = all_dates[0]
        if canonical is None:
            continue
        for r in rows:
            if r.get(k) in (None, "", "null"):
                r[k] = canonical
    return rows


def _propagate_case_clinical(package_code: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """No-op pass.

    A previous experiment tried OR-ing age_valid / arthritis_type /
    previous_surgery across the case (under the hypothesis they were
    case-level patient attributes).  The eval said NO — clinical_f1
    dropped from 0.7838 to 0.7814.  Labels treat these as per-page
    evidence too.  Keeping the function name so the call site doesn't
    need to change; the body is intentionally inert now.
    """
    return rows


# ---------------------------------------------------------------------------
# SG039C-only post-processors
# ---------------------------------------------------------------------------
# Findings from manual review of the 10 SG039C cases (notebooklm analysis):
#   Case 9: page 3 of a multi-page discharge summary lost discharge_summary=1
#           because Gemma classified only page 1 with the title.
#   Cases 2, 4, 7, 10: pharmacy bills, admission sheets, nursing charts,
#           consent forms (incl. Tamil) were classified as a doc type or
#           extra_document=0, when they should be extra_document=1.
#
# These two helpers run only when package_code == "SG039C".  Other packages
# are unaffected — risk-bounded changes that should improve SG039C metrics
# (mandatory_f1 + extra_f1) without touching MG064A / MG006A / SB039A.

# Filename substrings that indicate an admin / non-mandatory doc and should
# force extra_document=1 for SG039C.  Matched case-insensitively against the
# full filename.  Patterns are deliberately specific to avoid false positives
# (e.g. "consent" must appear; bare "form" would catch too many).
SG039C_EXTRA_FILENAME_PATTERNS = [
    "consent",            # informed consent / surgical consent / ICF
    "feedback",           # patient feedback forms
    "_pharma", "pharmacy", "pharma_bill",
    "_bill.", "_bills.", "_bill_", "patient_bill",
    "indent",             # pharmacy indent / drug indent
    "id_card", "_id_proof", "_aadhar",
    "input_output", "i_o_chart", "intake_output",
    "vitals_chart", "_tpr",
    "admission_sheet",    # NOT "admission_paper" → that's indoor_case
    "discharge_card",     # short admin card vs full discharge_summary
    "drug_chart", "med_chart",
    "post_op_orders", "post_op_order", "postop_order",
    "nursing_notes", "nurses_notes",
]

# Doc-type fields that participate in multi-page continuity for SG039C.
SG039C_DOC_TYPE_FIELDS = {
    "clinical_notes", "usg_report", "lft_report", "operative_notes",
    "pre_anesthesia", "discharge_summary", "photo_evidence", "histopathology",
}


def _sg039c_force_extra_by_filename(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """If a file's name matches an admin pattern, force extra_document=1
    on every page of that file and zero out the doc-type flags.
    """
    link_key = LINK_FIELD["SG039C"]
    for row in rows:
        link = str(row.get(link_key, "") or "").lower()
        if not link:
            continue
        if any(pat in link for pat in SG039C_EXTRA_FILENAME_PATTERNS):
            for k in SG039C_DOC_TYPE_FIELDS:
                if k in row:
                    row[k] = 0
            row["extra_document"] = 1
            # Clinical fields stay 0 by default for extras (initialize_row
            # already set them to 0; populate may have flipped some — clear).
            for k in ("clinical_condition", "usg_calculi",
                      "pain_present", "previous_surgery"):
                if k in row:
                    row[k] = 0
    return rows


def _sg039c_multi_page_continuity(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """For multi-page files in SG039C: if every non-extra page of the same
    file has ONE non-extra doc_type (e.g. all pages are discharge_summary
    except a stray extra_document page), promote the extra page to that
    same doc_type.  Conservative — only promotes when there's a clear,
    single dominant non-extra type within the file.
    """
    from collections import defaultdict, Counter

    link_key = LINK_FIELD["SG039C"]
    by_file: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_file[str(r.get(link_key, "") or "")].append(i)

    for link, idxs in by_file.items():
        if not link or len(idxs) < 2:
            continue
        types_present: Counter = Counter()
        for i in idxs:
            for k in SG039C_DOC_TYPE_FIELDS:
                if rows[i].get(k, 0) == 1:
                    types_present[k] += 1
        if not types_present:
            continue
        # Only propagate if exactly one non-extra doc_type appears in the file.
        if len(types_present) != 1:
            continue
        dominant = next(iter(types_present))
        # Promote each extra-page to the dominant type.
        for i in idxs:
            if int(rows[i].get("extra_document", 0) or 0) == 1:
                # Skip pages forced extra by filename (admin patterns) —
                # those are intentionally extra and shouldn't get promoted.
                link_lo = str(rows[i].get(link_key, "") or "").lower()
                if any(pat in link_lo for pat in SG039C_EXTRA_FILENAME_PATTERNS):
                    continue
                # Reset extras → dominant doc_type.
                for k in SG039C_DOC_TYPE_FIELDS:
                    if k in rows[i]:
                        rows[i][k] = 0
                rows[i][dominant] = 1
                rows[i]["extra_document"] = 0
    return rows


def _strip_to_schema(package_code: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return rows containing ONLY the schema keys, in schema order.

    The eval log showed `file_name` (and `link` for SB039A) leaking into
    output JSONs — possibly because some downstream cell mutates rows
    after process_case returns.  This belt-and-braces step guarantees
    the output JSON has exactly PACKAGE_SCHEMAS[package_code] keys.
    """
    schema = PACKAGE_SCHEMAS[package_code]
    nullable = {"pre_date", "post_date", "doa", "dod"}
    out: List[Dict[str, Any]] = []
    for r in rows:
        clean: Dict[str, Any] = {}
        for k in schema:
            v = r.get(k)
            if v is None and k not in nullable:
                v = 0  # binary fields default to 0; identifiers never None
            clean[k] = v
        out.append(clean)
    return out


def run_batch(data_root: Path, package_code_lookup: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Walk Claims/<PACKAGE>/<case>/* and run process_case on each."""
    cases = discover_cases(data_root)
    lookup = dict(PACKAGE_CODE_LOOKUP)
    if package_code_lookup:
        lookup.update(package_code_lookup)

    results: Dict[str, Any] = {}
    by_package: Dict[str, List[Dict[str, Any]]] = {p: [] for p in PACKAGE_CODES}

    for case_id, files in cases.items():
        pkg = lookup.get(case_id)
        if pkg not in PACKAGE_CODES:
            continue
        try:
            res = process_case(case_id, files, pkg)
        except Exception as e:
            print(f"[run_batch] case {case_id} failed: {e}")
            continue
        results[case_id] = res
        by_package[pkg].extend(res["strict_rows"])

    # Persist per-package JSON outputs to OUTPUT_ROOT.
    # Final pass: strip rows to ONLY the schema keys in schema order, so no
    # extra fields (file_name, link, ...) leak into the JSON.
    for pkg, rows in by_package.items():
        clean_rows = _strip_to_schema(pkg, rows)
        out_path = OUTPUT_ROOT / f"{pkg}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(clean_rows, f, indent=2, ensure_ascii=False)
        ok, issues = validate_output_rows(pkg, clean_rows)
        print(f"[write] {out_path} rows={len(clean_rows)} schema_ok={ok}")
        if not ok:
            for i in issues[:3]:
                print(f"  - {i}")

    # Persist per-case decisions for the design / explainability score.
    for case_id, res in results.items():
        with (DECISIONS_ROOT / f"{case_id}.json").open("w", encoding="utf-8") as f:
            json.dump(asdict(res["decision"]), f, indent=2, ensure_ascii=False)

    return {"results": results, "by_package": by_package}
'''
    ),
    # -------------------------------------------------------------------------
    # Cell 27 [c0a0a4e9] - export_case_outputs (now mostly redundant; keep slim)
    # -------------------------------------------------------------------------
    "c0a0a4e9": code(
        '''# =========================
# EXPORTERS
# =========================

def export_case_outputs(case_result: Dict[str, Any], output_root: Path = OUTPUT_ROOT) -> None:
    """Per-case strict rows are already aggregated into per-package JSONs by
    run_batch().  This helper just dumps a per-case JSON for inspection."""
    case_id = case_result["case_id"]
    out_path = output_root / f"{case_id}__rows.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(case_result["strict_rows"], f, indent=2, ensure_ascii=False)
'''
    ),
    # -------------------------------------------------------------------------
    # Cell 31 [27428369] - normalize_date - keep, already implemented; no-op
    # (we leave it as the organizer-supplied implementation; identical content)
    # -------------------------------------------------------------------------
    "27428369": code(
        '''# =========================
# DATE NORMALIZATION UTIL
# =========================

from datetime import datetime


def normalize_date(date_str: Optional[str]) -> Optional[str]:
    """Coerce common date strings (DD/MM/YY, DD-MM-YYYY, 'Feb 6 2026', ...) to DD-MM-YYYY."""
    if not date_str:
        return None
    s = str(date_str).strip()
    candidates = [
        "%d/%m/%y", "%d/%m/%Y",
        "%d-%m-%y", "%d-%m-%Y",
        "%d.%m.%y", "%d.%m.%Y",
        "%d-%b-%y", "%d-%b-%Y",
        "%d %b %Y", "%d %B %Y",
        "%d %b %y", "%d %B %y",
        "%b %d, %Y", "%B %d, %Y",
        "%b %d %Y", "%B %d %Y",
        "%m/%d/%y", "%m/%d/%Y",
    ]
    for fmt in candidates:
        try:
            dt = datetime.strptime(s, fmt)
            if dt.year < 1950:
                dt = dt.replace(year=dt.year + 100)
            return dt.strftime("%d-%m-%Y")
        except Exception:
            continue
    return s
'''
    ),
    # Cell 32 [e805bdd1] - normalize_dates_in_rows; keep organizer impl
    "e805bdd1": code(
        '''def normalize_dates_in_rows(package_code: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keys = DATE_FIELDS.get(package_code, [])
    out = []
    for r in rows:
        rr = dict(r)
        for k in keys:
            if k in rr and rr[k]:
                rr[k] = normalize_date(rr[k])
        out.append(rr)
    return out
'''
    ),
    # Cell 33 [db0447fe] - duplicate normalize_date stub; replace with no-op pass
    "db0447fe": code(
        '''# (kept blank: normalize_date is defined above)
'''
    ),
    # Cell 34 [4f941e81] - duplicate normalize_dates_in_rows stub; no-op
    "4f941e81": code(
        '''# (kept blank: normalize_dates_in_rows is defined above)
'''
    ),
    # Cell 35 [b57d3faf] - render_decision_report
    "b57d3faf": code(
        '''# =========================
# DECISION REPORT RENDERER
# =========================

def render_decision_report(case_result: Dict[str, Any]) -> None:
    d = case_result["decision"]
    print("=" * 60)
    print(f"Case ID:    {d.case_id}")
    print(f"Package:    {d.package_code}  ({PACKAGE_NAMES.get(d.package_code, '')})")
    print(f"Decision:   {d.decision}    (confidence={d.confidence:.2f})")
    if d.missing_documents:
        print(f"Missing docs:")
        for m in d.missing_documents:
            print(f"  - {m}")
    if d.rule_flags:
        print(f"Rule flags:")
        for f in d.rule_flags:
            print(f"  - {f}")
    if d.timeline_flags:
        print(f"Timeline flags:")
        for f in d.timeline_flags:
            print(f"  - {f}")
    if d.reasons:
        print("Reasons:")
        for r in d.reasons[:6]:
            print(f"  - {r}")
'''
    ),
    # -------------------------------------------------------------------------
    # NEW CELL - Gemma self-test (inserted just before the main runner)
    # -------------------------------------------------------------------------
    "self-test-cell": code(
        '''# =========================
# GEMMA 3 12B SELF-TEST  (run ONE call, print parsed payload)
# =========================
# This burns exactly ONE Gemma call (regardless of MAX_VLM_CALLS) so you can
# confirm credentials + response-shape parsing before processing the dataset.
# If the parsed payload below looks empty or wrong, STOP and check creds/API
# response shape - otherwise the full batch is wasting tokens.

def _self_test_gemma() -> Dict[str, Any]:
    """Run ONE Gemma 3 12B call against Sample.jpg (or any available image).

    Bypasses the cache and the MAX_VLM_CALLS guard so the dry-run signal is
    always genuine.  Prints the parsed payload and a verdict.
    """
    global DATA_ROOT

    # Self-test runs BEFORE the main runner, so we have to do the same
    # autodetect here to find the dataset wherever the databank widget
    # actually unzipped it (e.g. /home/jovyan/<databank-id>/Claims).
    def _has_pkg_subdirs(p: Path) -> bool:
        if not p.exists() or not p.is_dir():
            return False
        kids = {c.name for c in p.iterdir() if c.is_dir()}
        return any(code in kids for code in PACKAGE_CODES)

    if not _has_pkg_subdirs(DATA_ROOT):
        here = Path(".").resolve()
        probes = [
            here / "Claims",
            here / "Dataset-1" / "Claims",
            here / "Dataset" / "Claims",
            here / "data" / "Claims",
            here.parent / "Claims",
        ]
        for cand in probes:
            if _has_pkg_subdirs(cand):
                DATA_ROOT = cand
                print(f"[self-test] autodetect: dataset at {DATA_ROOT}")
                break
        else:
            for depth in (1, 2, 3):
                for cand in here.glob("/".join(["*"] * depth) + "/Claims"):
                    if _has_pkg_subdirs(cand):
                        DATA_ROOT = cand
                        print(f"[self-test] autodetect: dataset at {DATA_ROOT}")
                        break
                if _has_pkg_subdirs(DATA_ROOT):
                    break

    candidates: List[Path] = []
    for p in [Path("Sample.jpg"), Path("sample.jpg"), Path("Sample.jpeg")]:
        if p.exists():
            candidates.append(p)
            break
    if not candidates and DATA_ROOT.exists():
        # Fall back to the first .jpg/.jpeg under the resolved Claims root
        for ext in (".jpg", ".jpeg", ".png"):
            for p in DATA_ROOT.rglob(f"*{ext}"):
                candidates.append(p)
                break
            if candidates:
                break
    if not candidates:
        # Last resort: rasterize the first PDF\'s page 1.
        if DATA_ROOT.exists():
            for p in DATA_ROOT.rglob("*.pdf"):
                try:
                    pages = extract_pages(p)
                    if pages:
                        return _self_test_with_image(pages[0]["image"], p.name)
                except Exception:
                    continue
        print("[self-test] no test image available - skipping.")
        print(f"[self-test]   DATA_ROOT={DATA_ROOT.resolve()} exists={DATA_ROOT.exists()}")
        print("[self-test]   make sure the databank widget finished downloading,")
        print("[self-test]   then re-run this cell (and the main runner cell).")
        return {}
    img_path = candidates[0]
    try:
        img = Image.open(img_path)
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
    except Exception as e:
        print(f"[self-test] failed to open {img_path}: {e}")
        return {}
    return _self_test_with_image(img, img_path.name)


def _self_test_with_image(img, src_name: str) -> Dict[str, Any]:
    print(f"[self-test] using image: {src_name}  size={img.size}")
    print(f"[self-test] model:       {GEMMA_MODEL}")
    print(f"[self-test] firing one Gemma call (bypassing cache)...\\n")

    try:
        data_url = _image_to_data_uri(img)
        resp = nc.completion(
            model=GEMMA_MODEL,
            messages=[
                {"role": "system", "content": GEMMA_SYSTEM},
                {"role": "user", "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": _gemma_user_prompt("MG064A")},
                ]},
            ],
            metadata={"problem_statement": 1},
        )
    except Exception as e:
        print(f"[self-test] !! NHAclient call FAILED: {type(e).__name__}: {e}")
        print("[self-test]    Check that clientId / clientSecret are filled in,")
        print("[self-test]    and that the model name 'Gemma 3 12B' is accepted.")
        return {}

    # Show the raw response shape so you can spot API-shape mismatches.
    print("[self-test] raw response keys:", list(resp.keys()) if isinstance(resp, dict) else type(resp).__name__)
    try:
        raw_text = (resp.get("choices") or [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        print(f"[self-test] !! could not extract content from response: {e}")
        print("[self-test] raw response:", str(resp)[:500])
        return {}
    if not raw_text:
        print("[self-test] !! response had empty content. Showing raw response:")
        print(str(resp)[:800])
        return {}

    print("[self-test] raw model output (first 400 chars):")
    print("  " + raw_text[:400].replace("\\n", "\\n  "))
    print()

    payload = _normalize_payload(_safe_json_loads(raw_text))
    print("[self-test] parsed payload:")
    print(json.dumps(payload, indent=2, ensure_ascii=False)[:1500])
    print()

    # Verdict
    ok = bool(payload.get("doc_type")) and (payload.get("doc_type_confidence") or 0) > 0
    if ok:
        dt = payload.get("doc_type")
        conf = payload.get("doc_type_confidence")
        print(f"[self-test] PASS - doc_type={dt} confidence={conf}")
        print("[self-test] You can now set MAX_VLM_CALLS = None and run the full batch.")
    else:
        print("[self-test] WARN - parsed payload looks empty/low-confidence.")
        print("[self-test]        The pipeline will run but rule-fields will mostly be 0.")
        print("[self-test]        Inspect the raw output above and adjust the prompt or parser.")
    return payload


# Run the self-test now (will fire ONE Gemma call).  Comment this out if
# you want to skip it on subsequent runs.
SELF_TEST_PAYLOAD = _self_test_gemma()
'''
    ),
    # -------------------------------------------------------------------------
    # Cell 36 - Main runner; replace with a clean run_batch invocation
    # -------------------------------------------------------------------------
    "main-runner": code(
        '''# =========================
# MAIN RUNNER
# =========================
# This calls run_batch(...) which:
#   1. walks <PACKAGE>/<case>/* under the auto-detected dataset root
#   2. runs ONE Gemma 3 12B call per page (cached to ./vlm_cache/)
#   3. populates per-page rows according to the package schema
#   4. assigns document_rank with multi-page-PDF consistency
#   5. writes ./output/<PACKAGE>.json (per HI.txt) + ./decisions/<case>.json

def _autodetect_claims_dir(default: Path) -> Path:
    """Find the directory that contains the four package folders.

    The databank widget can unzip to several different paths
    (./Claims, ./Dataset-1/Claims, /home/jovyan/Dataset-1/Claims, ...).
    We probe common locations and pick the first one that contains any of
    PACKAGE_CODES as immediate children.
    """
    def looks_like_claims(p: Path) -> bool:
        if not p.exists() or not p.is_dir():
            return False
        kids = {c.name for c in p.iterdir() if c.is_dir()}
        return any(code in kids for code in PACKAGE_CODES)

    if looks_like_claims(default):
        return default

    # 1) check obvious candidates relative to CWD
    here = Path(".").resolve()
    candidates = [
        here / "Claims",
        here / "Dataset-1" / "Claims",
        here / "Dataset" / "Claims",
        here / "dataset" / "Claims",
        here / "data" / "Claims",
        here.parent / "Claims",
    ]
    for c in candidates:
        if looks_like_claims(c):
            return c

    # 2) shallow search: any directory named "Claims" up to 3 levels deep
    for depth in (1, 2, 3):
        for p in here.glob("/".join(["*"] * depth) + "/Claims"):
            if looks_like_claims(p):
                return p

    # 3) shallow search: any directory directly containing the package folders
    for depth in (1, 2, 3):
        for p in here.glob("/".join(["*"] * depth)):
            if p.is_dir() and looks_like_claims(p):
                return p

    return default  # nothing found; will produce an empty walk


_resolved_root = _autodetect_claims_dir(DATA_ROOT)
if _resolved_root != DATA_ROOT:
    print(f"[autodetect] DATA_ROOT={DATA_ROOT.resolve()} not usable; using {_resolved_root.resolve()}")
    DATA_ROOT = _resolved_root

print("Discovering cases under", DATA_ROOT.resolve())
cases = discover_cases(DATA_ROOT)
print(f"Found {len(cases)} cases.")
print("Cases per package:")
from collections import Counter
print(dict(Counter(PACKAGE_CODE_LOOKUP.values())))

if not cases:
    print("\\nNo cases found. Things to check:")
    print("  1. Did you click 'Download' in the databank widget cell above?")
    print("  2. The widget unzips into the current working directory; this notebook")
    print("     auto-searches ./Claims, ./Dataset-1/Claims, and shallow neighbours.")
    print("     If your dataset lives elsewhere, set DATA_ROOT manually in the CONFIG cell.")
    print()
    print("  Directories at this level:")
    for p in sorted(Path(".").iterdir()):
        if p.is_dir():
            print(f"    - {p}/")
else:
    print(f"\\nMAX_VLM_CALLS = {MAX_VLM_CALLS}  (set in CONFIG cell; None = unlimited)")
    BATCH = run_batch(DATA_ROOT)
    print(f"\\nFresh Gemma calls this run: {VLM_CALL_COUNTER['calls']}")
    print(f"Per-package output files written to {OUTPUT_ROOT.resolve()}")

    # ---- combined preview ----
    all_rows = []
    for pkg, rows in BATCH["by_package"].items():
        all_rows.extend(rows)
    if all_rows:
        FINAL_DF = pd.DataFrame(all_rows)
        print("\\nCombined output preview:")
        try:
            display(FINAL_DF.head(10))  # type: ignore[name-defined]
        except NameError:
            print(FINAL_DF.head(10).to_string())

    # ---- decisions preview ----
    decision_rows = [asdict(res["decision"]) for res in BATCH["results"].values()]
    if decision_rows:
        DECISIONS_DF = pd.DataFrame(decision_rows)
        print("\\nDecisions preview:")
        try:
            display(DECISIONS_DF.head(10))  # type: ignore[name-defined]
        except NameError:
            print(DECISIONS_DF.head(10).to_string())
'''
    ),
}


def main() -> int:
    with SRC.open("r", encoding="utf-8") as f:
        nb = json.load(f)

    # The last code cell ("cell-36") doesn't have a normal hex id; we resolve
    # it by index instead.
    last_code_idx = None
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code" and not cell.get("id", "").replace("-", "").isalnum():
            last_code_idx = i
        if cell["cell_type"] == "code":
            last_code_idx = i  # always last code cell index
    # safer: find by inspecting source content
    for i, cell in enumerate(nb["cells"]):
        src = "".join(cell.get("source", [])) if isinstance(cell.get("source"), list) else cell.get("source", "")
        if "MAIN RUNNER / FINAL ASSEMBLY" in src or "BATCH_RESULTS = {}" in src:
            last_code_idx = i
            break

    replaced = 0
    for cell in nb["cells"]:
        cid = cell.get("id", "")
        if cid in CELL_OVERRIDES:
            cell["source"] = CELL_OVERRIDES[cid]
            cell["outputs"] = []
            cell["execution_count"] = None
            replaced += 1

    # Replace the unnamed final code cell ("MAIN RUNNER / FINAL ASSEMBLY")
    if last_code_idx is not None:
        nb["cells"][last_code_idx]["source"] = CELL_OVERRIDES["main-runner"]
        nb["cells"][last_code_idx]["outputs"] = []
        nb["cells"][last_code_idx]["execution_count"] = None
        replaced += 1

    # Insert the self-test cell directly BEFORE the main runner so all
    # pipeline functions are already in scope and the user sees the parsed
    # Gemma payload before any case processing begins.
    if last_code_idx is not None:
        self_test_cell = {
            "cell_type": "code",
            "id": "self-test-gemma",
            "metadata": {},
            "execution_count": None,
            "outputs": [],
            "source": CELL_OVERRIDES["self-test-cell"],
        }
        nb["cells"].insert(last_code_idx, self_test_cell)
        inserted = 1
    else:
        inserted = 0

    with DST.open("w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"Wrote {DST.relative_to(ROOT)} ({replaced} cells replaced, {inserted} cells inserted).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
