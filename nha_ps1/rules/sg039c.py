"""Cholecystectomy (SG039C) — STG fields.

HI.txt §B.3-B.4:
  clinical_condition  -> any of 5 STG conditions present (Page 2 §1.2)
  usg_calculi         -> USG shows calculi
  pain_present        -> pain documented (right hypochondrium / epigastrium)
  previous_surgery    -> prior cholecystectomy / abdominal surgery (fraud check)
"""

from __future__ import annotations

from typing import Any, Dict

from ..extract import contains_any

CLINICAL_CONDITIONS = [
    "cholelithiasis", "cholecystitis", "biliary colic",
    "gallstone", "gall stone", "gall bladder stone",
    "acute cholecystitis", "chronic cholecystitis",
    "biliary pancreatitis", "obstructive jaundice",
]
USG_CALCULI_TERMS = [
    "calculi", "calculus", "stones", "cholelithiasis",
    "echogenic focus", "gall bladder calculi", "gb calculi",
    "multiple calculi", "single calculus",
]
PAIN_TERMS = [
    "abdominal pain", "right hypochondrium", "epigastrium", "epigastric pain",
    "right upper quadrant", "ruq pain", "biliary colic", "pain abdomen",
]
PREVIOUS_SURGERY_TERMS = [
    "previous surgery", "prior surgery", "history of surgery",
    "previous cholecystectomy", "prior cholecystectomy",
    "post cholecystectomy", "h/o surgery", "previous laparotomy",
]


def apply(row: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    case_text = ctx.get("case_text_en") or ctx.get("case_text") or ""
    row["clinical_condition"] = contains_any(case_text, CLINICAL_CONDITIONS)
    row["usg_calculi"] = contains_any(case_text, USG_CALCULI_TERMS)
    row["pain_present"] = contains_any(case_text, PAIN_TERMS)
    row["previous_surgery"] = contains_any(case_text, PREVIOUS_SURGERY_TERMS)
    return row
