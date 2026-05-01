"""Total Knee Replacement (SB039A) — STG fields.

HI.txt §D.2-D.4:
  doa, dod                 -> from discharge summary, DD-MM-YYYY
  arthritis_type           -> Page 1-2 §1.2 — any condition justifies TKR
  post_op_implant_present  -> Page 4 §3.2.2 — implant visible in post-op imaging
  age_valid                -> Page 4 §3.2.3 — age threshold (typically >55)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from ..extract import (
    contains_any,
    find_age,
    find_dates,
    first_or_none,
    normalize_date,
)

ARTHRITIS_TERMS = [
    "osteoarthritis", "rheumatoid arthritis", "primary osteoarthritis",
    "tricompartmental osteoarthritis", "tricompartmental oa",
    "bilateral oa", "secondary oa", "post traumatic arthritis",
    "post-traumatic arthritis", "avascular necrosis", "avn",
    "psoriatic arthritis", "gouty arthritis",
]
AGE_THRESHOLD = 55  # HI.txt §D.4 references age validation; full(1).py used >55.

DOA_PATTERNS = [
    r"date of admission[:\s]*([\d/.\- A-Za-z,]+?)(?:\s|$|date)",
    r"admission date[:\s]*([\d/.\- A-Za-z,]+?)(?:\s|$|date)",
    r"\bdoa[:\s]*([\d/.\- A-Za-z,]+?)(?:\s|$|date)",
    r"admitted on[:\s]*([\d/.\- A-Za-z,]+?)(?:\s|$|date)",
]
DOD_PATTERNS = [
    r"date of discharge[:\s]*([\d/.\- A-Za-z,]+?)(?:\s|$|date)",
    r"discharge date[:\s]*([\d/.\- A-Za-z,]+?)(?:\s|$|date)",
    r"\bdod[:\s]*([\d/.\- A-Za-z,]+?)(?:\s|$|date)",
    r"discharged on[:\s]*([\d/.\- A-Za-z,]+?)(?:\s|$|date)",
]


def _pattern_date(text: str, patterns: List[str]) -> Optional[str]:
    if not text:
        return None
    lo = text.lower()
    for p in patterns:
        m = re.search(p, lo)
        if m:
            cand = m.group(1).strip(" :,")
            for d in find_dates(cand):
                return normalize_date(d)
    return None


def _fallback_dates(dates: List[str]) -> tuple[Optional[str], Optional[str]]:
    """If pattern lookup fails, fall back to first/last date heuristically."""
    if not dates:
        return (None, None)
    if len(dates) == 1:
        return (normalize_date(dates[0]), None)
    return (normalize_date(dates[0]), normalize_date(dates[-1]))


def apply(row: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    text_en = ctx.get("text_en") or ctx.get("text") or ""
    case_text = ctx.get("case_text_en") or ctx.get("case_text") or text_en
    doc_type = ctx.get("doc_type", "")
    visual_tags = ctx.get("visual_tags", {}) or {}
    extracted = ctx.get("extracted", {}) or {}

    # Dates: only fill from discharge_summary or operative_notes pages.
    if doc_type in {"discharge_summary", "indoor_case", "clinical_notes"}:
        doa = extracted.get("doa") or _pattern_date(text_en, DOA_PATTERNS)
        dod = extracted.get("dod") or _pattern_date(text_en, DOD_PATTERNS)
        if not doa or not dod:
            fallback_dates = list(extracted.get("dates_found") or []) or find_dates(text_en)
            fb_doa, fb_dod = _fallback_dates(fallback_dates)
            doa = doa or fb_doa
            dod = dod or fb_dod
        row["doa"] = normalize_date(doa)
        row["dod"] = normalize_date(dod)

    row["arthritis_type"] = contains_any(case_text, ARTHRITIS_TERMS)

    # post_op_implant_present: implant sticker OR x-ray on a post-op X-ray page.
    has_implant = int(bool(visual_tags.get("has_implant_sticker")))
    if doc_type == "post_op_xray" and visual_tags.get("has_xray"):
        has_implant = 1
    if contains_any(case_text, ["implant in situ", "prosthesis in place", "knee prosthesis"]):
        has_implant = 1
    row["post_op_implant_present"] = has_implant

    # age_valid: extracted age >= threshold (>55 in full(1).py).
    age = extracted.get("age") if isinstance(extracted.get("age"), int) else find_age(case_text)
    row["age_valid"] = int(bool(age and age > AGE_THRESHOLD))

    return row
