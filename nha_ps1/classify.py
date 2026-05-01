"""Document-type classification.

Primary path: trust the VLM's `doc_type` and `doc_type_confidence` fields.
Fallback (low-VLM-confidence or empty OCR): keyword heuristics ported from
full(1).py with visual-tag boosts.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

from .schemas import DOCUMENT_TYPES

CONFIDENCE_FALLBACK_THRESHOLD = 0.40

DOC_KEYWORDS: Dict[str, list] = {
    "clinical_notes": [
        "clinical", "examination", "history", "complaint", "diagnosis",
        "assessment", "ho/", "h/o", "presenting complaint", "general examination",
    ],
    "discharge_summary": [
        "discharge summary", "discharge", "final diagnosis", "discharged",
        "date of discharge", "condition at discharge",
    ],
    "cbc_hb_report": [
        "cbc", "complete blood count", "haemogram", "hemogram",
        "hemoglobin", "haemoglobin", "rbc", "wbc", "platelet",
    ],
    "indoor_case": [
        "indoor", "admission record", "admitted", "ipd", "ward", "bed no",
        "in-patient", "inpatient",
    ],
    "treatment_details": [
        "treatment", "transfusion", "ferrous sulphate", "medication chart",
        "drug administration", "blood unit", "infusion",
    ],
    "post_hb_report": [
        "post hb", "post-treatment hb", "post transfusion", "repeat hb",
        "follow up hb",
    ],
    "usg_report": [
        "ultrasound", "usg", "sonography", "gallbladder", "abdomen",
    ],
    "lft_report": [
        "liver function", "lft", "bilirubin", "sgot", "sgpt", "alt", "ast",
        "alkaline phosphatase",
    ],
    "operative_notes": [
        "operative notes", "ot notes", "operation notes", "surgery", "procedure performed",
        "intra-op", "intra op", "surgeon", "anaesthetist",
    ],
    "pre_anesthesia": [
        "pre-anesthesia", "pre anesthesia", "pac", "anaesthesia fitness",
        "anesthesia clearance", "anaesthesia checkup",
    ],
    "histopathology": [
        "histopathology", "biopsy", "microscopic examination", "histology",
    ],
    "xray_ct_knee": [
        "x-ray knee", "xray knee", "ct knee", "knee joint", "knee radiograph",
    ],
    "investigation_pre": [
        "widal", "typhi", "blood culture", "investigation report", "lab report",
    ],
    "investigation_post": [
        "repeat widal", "post treatment", "follow up investigation",
    ],
    "vitals_treatment": [
        "vitals", "temperature chart", "tpr", "pulse rate", "blood pressure",
    ],
    "implant_invoice": [
        "invoice", "tax invoice", "implant", "prosthesis", "bill",
        "barcode", "lot no", "ref no",
    ],
    "post_op_photo": [
        "post operative photo", "post-op photo", "patient photograph",
    ],
    "post_op_xray": [
        "post operative x-ray", "post-op x-ray", "post op xray",
    ],
    "photo_evidence": [
        "intraoperative photograph", "intra operative photo", "specimen photo",
    ],
}

VISUAL_BOOSTS = {
    "implant_invoice": [("has_table", 3), ("has_implant_sticker", 5), ("has_barcode", 4)],
    "post_op_photo": [("has_photo_evidence", 5)],
    "photo_evidence": [("has_photo_evidence", 5)],
    "operative_notes": [("has_signature", 2), ("has_stamp", 1)],
    "discharge_summary": [("has_signature", 2), ("has_stamp", 2)],
    "post_op_xray": [("has_xray", 5)],
    "xray_ct_knee": [("has_xray", 5)],
    "usg_report": [("has_xray", 2)],
}


def keyword_classify(text: str, visual_tags: Dict[str, int]) -> Tuple[str, float]:
    if not text:
        return ("extra_document", 0.0)
    lo = text.lower()
    scores: Dict[str, int] = {}
    for dt, kws in DOC_KEYWORDS.items():
        s = sum(1 for kw in kws if kw in lo)
        for vk, boost in VISUAL_BOOSTS.get(dt, []):
            if visual_tags.get(vk):
                s += boost
        scores[dt] = s

    if not scores or max(scores.values()) <= 0:
        return ("extra_document", 0.20)
    best = max(scores, key=scores.get)
    confidence = min(scores[best] / 8.0, 0.85)
    return (best, confidence)


def resolve_doc_type(vlm_payload: Dict[str, Any], visual_tags: Dict[str, int]) -> Tuple[str, float]:
    """Pick the final doc_type using VLM-first with keyword fallback."""
    vlm_type = str(vlm_payload.get("doc_type", "extra_document")).lower()
    vlm_conf = float(vlm_payload.get("doc_type_confidence", 0.0) or 0.0)

    if vlm_type in DOCUMENT_TYPES and vlm_conf >= CONFIDENCE_FALLBACK_THRESHOLD:
        return (vlm_type, vlm_conf)

    text = str(vlm_payload.get("ocr_text_en") or vlm_payload.get("ocr_text") or "")
    kw_type, kw_conf = keyword_classify(text, visual_tags)

    # Prefer the higher-confidence answer; if VLM picked a known type with
    # weak confidence and the keyword path agrees, bump confidence.
    if vlm_type == kw_type and vlm_type in DOCUMENT_TYPES:
        return (vlm_type, max(vlm_conf, kw_conf))
    if kw_conf >= vlm_conf:
        return (kw_type, kw_conf)
    return (vlm_type if vlm_type in DOCUMENT_TYPES else "extra_document", vlm_conf)
