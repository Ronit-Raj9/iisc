"""Output schemas, document-type taxonomy, rank maps, and mandatory-doc lists.

The PACKAGE_SCHEMAS keys follow HI.txt example JSONs verbatim:
- MG064A uses 'link'
- SG039C uses 'S3_link/DocumentName'
- MG006A uses 'S3_link'
- SB039A uses 's3_link'

HI.txt: "Solutions that do not strictly adhere to the specified output format
will be rejected without evaluation." -> we honor each spelling exactly.
"""

from __future__ import annotations

from typing import Dict, List

PACKAGE_SCHEMAS: Dict[str, List[str]] = {
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

# Per-package name of the document-link field, derived from PACKAGE_SCHEMAS.
LINK_FIELD: Dict[str, str] = {
    "MG064A": "link",
    "SG039C": "S3_link/DocumentName",
    "MG006A": "S3_link",
    "SB039A": "s3_link",
}

# Date fields per package (DD-MM-YYYY normalization downstream).
DATE_FIELDS: Dict[str, List[str]] = {
    "MG064A": [],
    "SG039C": [],
    "MG006A": ["pre_date", "post_date"],
    "SB039A": ["doa", "dod"],
}

# Closed taxonomy of document types we recognize across packages.
DOCUMENT_TYPES: List[str] = [
    "clinical_notes",
    "cbc_hb_report",
    "indoor_case",
    "treatment_details",
    "post_hb_report",
    "discharge_summary",
    "usg_report",
    "lft_report",
    "operative_notes",
    "pre_anesthesia",
    "histopathology",
    "xray_ct_knee",
    "investigation_pre",
    "investigation_post",
    "vitals_treatment",
    "implant_invoice",
    "post_op_photo",
    "post_op_xray",
    "photo_evidence",
    "extra_document",
]

# Mandatory documents per package (notebook cell 24 + HI.txt §A.2 / B.2 / C.2 / D.2).
MANDATORY_DOCS: Dict[str, List[str]] = {
    "MG064A": [
        "clinical_notes", "cbc_hb_report", "indoor_case",
        "treatment_details", "post_hb_report", "discharge_summary",
    ],
    "SG039C": [
        "clinical_notes", "usg_report", "lft_report", "operative_notes",
        "pre_anesthesia", "discharge_summary", "photo_evidence", "histopathology",
    ],
    "MG006A": [
        "clinical_notes", "investigation_pre", "vitals_treatment",
        "investigation_post", "discharge_summary",
    ],
    "SB039A": [
        "clinical_notes", "xray_ct_knee", "indoor_case", "operative_notes",
        "implant_invoice", "post_op_photo", "post_op_xray", "discharge_summary",
    ],
}

# Document timeline ranks per package (HI.txt §A.6 typical order).
RANK_MAP: Dict[str, Dict[str, int]] = {
    "MG064A": {
        "clinical_notes": 1,
        "cbc_hb_report": 2,
        "indoor_case": 2,
        "treatment_details": 3,
        "post_hb_report": 4,
        "discharge_summary": 5,
    },
    "SG039C": {
        "clinical_notes": 1,
        "usg_report": 2,
        "lft_report": 3,
        "pre_anesthesia": 4,
        "operative_notes": 5,
        "discharge_summary": 5,
        "histopathology": 6,
        "photo_evidence": 6,
    },
    "MG006A": {
        "clinical_notes": 1,
        "investigation_pre": 2,
        "vitals_treatment": 3,
        "investigation_post": 4,
        "discharge_summary": 5,
    },
    "SB039A": {
        "clinical_notes": 1,
        "xray_ct_knee": 2,
        "indoor_case": 3,
        "operative_notes": 4,
        "implant_invoice": 5,
        "post_op_photo": 6,
        "post_op_xray": 6,
        "discharge_summary": 7,
    },
}

# Non-binary keys that should NOT default to 0 when initializing a row.
NON_BINARY_KEYS = {
    "case_id", "link", "S3_link", "S3_link/DocumentName", "s3_link",
    "procedure_code", "page_number", "pre_date", "post_date", "doa", "dod",
    "document_rank",
}

# Date-typed keys (None default).
NULLABLE_DATE_KEYS = {"pre_date", "post_date", "doa", "dod"}
