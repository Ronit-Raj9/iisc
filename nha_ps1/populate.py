"""Build a single output row per page, matching PACKAGE_SCHEMAS exactly.

Row construction is two-stage:
  1. initialize with the schema's keys in the right order (case_id/link/etc.)
  2. set per-doc-type presence flag, then run the package's rule module to
     fill clinical-condition / signs / dates / age etc.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .rules import PACKAGE_RULE_FNS
from .schemas import (
    DOCUMENT_TYPES,
    LINK_FIELD,
    NULLABLE_DATE_KEYS,
    PACKAGE_SCHEMAS,
)


def initialize_row(
    package_code: str,
    case_id: str,
    file_name: str,
    page_number: int,
) -> Dict[str, Any]:
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


def populate_row(
    package_code: str,
    case_id: str,
    file_name: str,
    page_number: int,
    doc_type: str,
    visual_tags: Dict[str, int],
    quality: Dict[str, Any],
    text: str,
    text_en: str,
    extracted: Dict[str, Any],
    case_text: str,
    case_text_en: str,
) -> Dict[str, Any]:
    row = initialize_row(package_code, case_id, file_name, page_number)

    # Mark the doc-type presence flag if the schema has that key.
    if doc_type in row:
        row[doc_type] = 1

    # Mandatory: extra_document = 1 if doc_type unknown or 'extra_document'.
    if doc_type not in DOCUMENT_TYPES or doc_type == "extra_document":
        row["extra_document"] = 1

    ctx = {
        "doc_type": doc_type,
        "visual_tags": visual_tags or {},
        "quality": quality or {},
        "text": text or "",
        "text_en": text_en or text or "",
        "extracted": extracted or {},
        "case_text": case_text or "",
        "case_text_en": case_text_en or case_text or "",
    }
    rule_fn = PACKAGE_RULE_FNS[package_code]
    row = rule_fn(row, ctx)
    return row
