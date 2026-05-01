"""Severe Anemia (MG064A) — STG fields.

HI.txt §A.3-A.4:
  severe_anemia            -> Hb < 7 g/dL or explicit "severe anemia"
  common_signs             -> pallor, fatigue, weakness, etc. (Page 1-2)
  significant_signs        -> tachycardia, breathlessness, etc. (Page 2)
  life_threatening_signs   -> cardiac failure, severe hypoxia, shock (Page 2)
"""

from __future__ import annotations

from typing import Any, Dict

from ..extract import contains_any, parse_hb_values

COMMON_SIGNS = [
    "pallor", "fatigue", "weakness", "tiredness", "pale",
    "lethargy", "easily fatigued", "loss of appetite",
]
SIGNIFICANT_SIGNS = [
    "breathlessness", "shortness of breath", "dyspnea", "dyspnoea",
    "tachycardia", "palpitation", "palpitations", "dizziness",
    "syncope", "exertional dyspnea",
]
LIFE_THREATENING_SIGNS = [
    "cardiac failure", "heart failure", "chf", "severe hypoxia",
    "shock", "altered sensorium", "unconscious", "respiratory distress",
    "haemodynamic instability", "hemodynamic instability", "hemorrhagic shock",
]
SEVERITY_TEXT_TRIGGERS = [
    "severe anemia", "severe anaemia",
    "hb<7", "hb < 7", "hb less than 7",
    "haemoglobin < 7", "hemoglobin < 7",
]


def apply(row: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    text_en = ctx.get("text_en") or ctx.get("text") or ""
    case_text = ctx.get("case_text_en") or ctx.get("case_text") or text_en

    # severe_anemia: Hb < 7 g/dL on any reading OR explicit phrase anywhere in case.
    hb_values = parse_hb_values(case_text)
    severe = 0
    if any(v < 7.0 for v in hb_values):
        severe = 1
    if contains_any(case_text, SEVERITY_TEXT_TRIGGERS):
        severe = 1
    row["severe_anemia"] = severe

    row["common_signs"] = contains_any(case_text, COMMON_SIGNS)
    row["significant_signs"] = contains_any(case_text, SIGNIFICANT_SIGNS)
    row["life_threatening_signs"] = contains_any(case_text, LIFE_THREATENING_SIGNS)

    return row
