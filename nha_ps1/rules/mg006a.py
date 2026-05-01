"""Enteric Fever (MG006A) — STG fields.

HI.txt §C.2-C.4:
  pre_date / post_date -> dates from pre/post investigation reports (DD-MM-YYYY)
  poor_quality         -> from page-quality estimator
  fever                -> Page 4 §3.2.1
  symptoms             -> Page 2 §1.2 (any of 3)
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..extract import contains_any, find_dates, first_or_none, normalize_date

FEVER_TERMS = [
    "fever", "pyrexia", "febrile", "high temperature",
    "temperature 100", "temperature 101", "temperature 102",
    "temp 100", "temp 101", "temp 102", "38.0", "38.5", "39.0", "39.5", "40",
    "raised temperature", "high grade fever",
]
ENTERIC_SYMPTOMS = [
    "headache", "abdominal pain", "diarrhea", "diarrhoea",
    "constipation", "vomiting", "nausea", "malaise",
    "loss of appetite", "rose spots", "hepatomegaly", "splenomegaly",
]


def _select_date(dates: List[str]) -> str | None:
    """Pick the first non-empty date and normalize to DD-MM-YYYY."""
    return normalize_date(first_or_none(dates))


def apply(row: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    text_en = ctx.get("text_en") or ctx.get("text") or ""
    case_text = ctx.get("case_text_en") or ctx.get("case_text") or text_en
    doc_type = ctx.get("doc_type", "")
    quality = ctx.get("quality", {})
    extracted = ctx.get("extracted", {}) or {}

    # Date extraction: only from the actual investigation pages per HI.txt §C.2.
    dates_in_doc = list(extracted.get("dates_found") or []) or find_dates(text_en)

    if doc_type == "investigation_pre":
        row["pre_date"] = _select_date(dates_in_doc)
    elif doc_type == "investigation_post":
        row["post_date"] = _select_date(dates_in_doc)

    row["poor_quality"] = int(bool(quality.get("is_poor_quality")))
    row["fever"] = contains_any(case_text, FEVER_TERMS)
    row["symptoms"] = contains_any(case_text, ENTERIC_SYMPTOMS)
    return row
