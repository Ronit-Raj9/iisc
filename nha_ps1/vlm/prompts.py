"""Master prompt for one-shot page analysis.

A single VLM call returns OCR + classification + visual flags + entity extraction
as one JSON object, validated with Pydantic. This is the design choice that
makes 434-page CPU inference tractable.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# ---------- Pydantic schema (constrained output) ------------------------------

class VisualElements(BaseModel):
    has_stamp: int = 0
    has_signature: int = 0
    has_photo_evidence: int = 0
    has_implant_sticker: int = 0
    has_table: int = 0
    has_xray: int = 0


class ExtractedEntities(BaseModel):
    patient_name: Optional[str] = None
    age: Optional[int] = None
    dates_found: List[str] = Field(default_factory=list)
    hb_values: List[float] = Field(default_factory=list)
    diagnoses: List[str] = Field(default_factory=list)
    symptoms_mentioned: List[str] = Field(default_factory=list)
    doa: Optional[str] = None
    dod: Optional[str] = None


class PageQuality(BaseModel):
    is_blurry: int = 0
    is_multilingual: int = 0
    language: str = "en"


class PageAnalysis(BaseModel):
    ocr_text: str = ""
    ocr_text_en: str = ""
    doc_type: str = "extra_document"
    doc_type_confidence: float = 0.0
    visual_elements: VisualElements = Field(default_factory=VisualElements)
    extracted: ExtractedEntities = Field(default_factory=ExtractedEntities)
    page_quality: PageQuality = Field(default_factory=PageQuality)


DOC_TYPE_VOCAB = [
    "clinical_notes", "cbc_hb_report", "indoor_case", "treatment_details",
    "post_hb_report", "discharge_summary", "usg_report", "lft_report",
    "operative_notes", "pre_anesthesia", "histopathology", "xray_ct_knee",
    "investigation_pre", "investigation_post", "vitals_treatment",
    "implant_invoice", "post_op_photo", "post_op_xray", "photo_evidence",
    "extra_document",
]


# ---------- The prompt --------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a medical-document analyst for the Indian Ayushman Bharat PMJAY "
    "claim adjudication system. You must read healthcare scans, perform OCR "
    "(including Hindi, Bengali, Assamese, Gujarati where present), classify "
    "each page into one of a fixed taxonomy, detect visual evidence (stamps, "
    "signatures, photos, implant stickers, tables, x-ray imagery), and extract "
    "structured entities. Reply with one JSON object and nothing else — no "
    "prose, no markdown, no code fences."
)


def build_user_prompt(package_code: str) -> str:
    types = ", ".join(DOC_TYPE_VOCAB)
    return f"""Analyze the attached page from a PMJAY claim under package {package_code}.

Return ONE JSON object exactly matching this schema (use 0/1 for booleans):

{{
  "ocr_text": "<full text in original language(s), preserving line breaks>",
  "ocr_text_en": "<English translation if not English; else copy of ocr_text>",
  "doc_type": "<one of: {types}>",
  "doc_type_confidence": 0.0,
  "visual_elements": {{
    "has_stamp": 0, "has_signature": 0, "has_photo_evidence": 0,
    "has_implant_sticker": 0, "has_table": 0, "has_xray": 0
  }},
  "extracted": {{
    "patient_name": null,
    "age": null,
    "dates_found": [],
    "hb_values": [],
    "diagnoses": [],
    "symptoms_mentioned": [],
    "doa": null,
    "dod": null
  }},
  "page_quality": {{
    "is_blurry": 0,
    "is_multilingual": 0,
    "language": "en"
  }}
}}

Classification rules:
- clinical_notes: doctor's history/examination/diagnosis notes (NOT discharge).
- cbc_hb_report: CBC or hemoglobin lab report (only for MG064A pre-treatment).
- post_hb_report: hemoglobin report taken AFTER treatment / transfusion.
- indoor_case: admission paper / IPD entry / ward record.
- treatment_details: medication chart, transfusion record, drug administration.
- discharge_summary: final summary on discharge with admission and discharge dates.
- usg_report: ultrasound (gallbladder/abdomen for SG039C).
- lft_report: liver function tests.
- operative_notes: surgical / OT notes describing the procedure.
- pre_anesthesia: PAC / anesthesia fitness clearance.
- histopathology: biopsy / microscopic pathology report.
- photo_evidence: intra-operative or clinical patient photograph.
- xray_ct_knee: pre-op X-ray or CT of the knee (SB039A).
- post_op_xray: X-ray taken after surgery (typically shows implant).
- post_op_photo: photograph of patient after surgery.
- implant_invoice: invoice/bill/sticker showing the implant used.
- investigation_pre: pre-treatment lab investigation (MG006A, e.g. Widal/CBC).
- investigation_post: post-treatment repeat investigation.
- vitals_treatment: vitals/temperature charts during treatment.
- extra_document: anything not in the above categories (consent, ID, unrelated).

Entity rules:
- dates_found: every date you can read, copied verbatim.
- doa / dod: only fill from a discharge summary (admission / discharge dates).
- hb_values: numeric Hb readings in g/dL only.
- symptoms_mentioned: e.g. ["pallor","fatigue","fever","abdominal pain"].

Output ONLY the JSON object."""


def few_shot_block(examples: list) -> str:
    """Render in-context examples as a string. `examples` is a list of dicts
    with keys `description` and `json`."""
    if not examples:
        return ""
    chunks = ["", "Reference examples (do not copy values literally):"]
    for ex in examples:
        chunks.append(f"- {ex.get('description', '')}: {ex.get('json', '')}")
    chunks.append("")
    return "\n".join(chunks)
