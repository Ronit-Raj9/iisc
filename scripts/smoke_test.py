"""End-to-end smoke test that does not require the VLM model.

Synthesizes a tiny fake VLM cache for one MG064A case, runs the pipeline with
backend=cache_only, and validates the output JSON. Useful as a CI canary.

    python scripts/smoke_test.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nha_ps1.config import SYNTHETIC_LABELS_ROOT
from nha_ps1.populate import populate_row
from nha_ps1.rank import assign_ranks
from nha_ps1.validate import validate_rows


def _build_case_rows(package, case_id, pages):
    """Mirror pipeline.process_case: join text across pages, then populate each row."""
    case_text = "\n".join(p["text"] for p in pages)
    rows = []
    for p in pages:
        rows.append(populate_row(
            package_code=package,
            case_id=case_id,
            file_name=p["file_name"],
            page_number=p.get("page_number", 1),
            doc_type=p["doc_type"],
            visual_tags=p.get("visual_tags", {}),
            quality=p.get("quality", {"is_poor_quality": False}),
            text=p["text"],
            text_en=p["text"],
            extracted=p.get("extracted", {}),
            case_text=case_text,
            case_text_en=case_text,
        ))
    return rows


def _p(file_name, doc_type, text, page_number=1, **extras):
    return {"file_name": file_name, "doc_type": doc_type, "text": text,
            "page_number": page_number, **extras}


def test_mg064a_severe_anemia_case():
    """A canonical severe-anemia case should mark the right flags."""
    pages = [
        _p("clinical.pdf", "clinical_notes",
           "Patient with pallor and fatigue, breathlessness on exertion. "
           "Diagnosis: severe anemia."),
        _p("cbc.pdf", "cbc_hb_report",
           "CBC report\nHb 5.4 g/dL\nWBC 7000\nPlatelet 220000",
           extracted={"hb_values": [5.4]}),
        _p("indoor.pdf", "indoor_case",
           "Patient admitted to medicine ward, IPD bed 12."),
        _p("treatment.pdf", "treatment_details",
           "Blood transfusion 1 unit. Ferrous sulphate IV."),
        _p("post_hb.pdf", "post_hb_report",
           "Repeat Hb after transfusion: 9.6 g/dL"),
        _p("discharge.pdf", "discharge_summary",
           "Patient discharged in stable condition."),
        _p("consent.pdf", "extra_document",
           "Consent form for blood transfusion."),
    ]
    rows = _build_case_rows("MG064A", "C1", pages)
    rows = assign_ranks("MG064A", rows)
    ok, issues = validate_rows("MG064A", rows)
    assert ok, f"Schema violations: {issues}"

    # Severe anemia should be flagged on every row (case-level signal).
    assert all(r["severe_anemia"] == 1 for r in rows), "severe_anemia should propagate via case_text"
    # The CBC row should be present.
    cbc = next(r for r in rows if r["link"] == "cbc.pdf")
    assert cbc["cbc_hb_report"] == 1
    assert cbc["document_rank"] == 2  # per RANK_MAP
    # The consent form is extra.
    consent = next(r for r in rows if r["link"] == "consent.pdf")
    assert consent["extra_document"] == 1
    assert consent["document_rank"] == 99
    # Discharge rank should be 5.
    dis = next(r for r in rows if r["link"] == "discharge.pdf")
    assert dis["document_rank"] == 5
    print("[OK] MG064A severe anemia case")


def test_multipage_pdf_consistency():
    """Pages of the same PDF must share document_rank (HI.txt §A.6 note)."""
    pages = [
        _p("clinical_combined.pdf", "clinical_notes",
           "Patient with right hypochondrium pain. Diagnosed with cholelithiasis.",
           page_number=1),
        # Page 2 of the same PDF classifies as 'extra_document' but should
        # INHERIT rank=1 from the dominant rank in the group.
        _p("clinical_combined.pdf", "extra_document",
           "Continuation of clinical notes with more findings.",
           page_number=2),
    ]
    rows = _build_case_rows("SG039C", "C2", pages)
    rows = assign_ranks("SG039C", rows)
    ranks = {r["page_number"]: r["document_rank"] for r in rows}
    assert ranks[1] == ranks[2], f"Pages of one PDF must share rank, got {ranks}"
    print(f"[OK] multi-page PDF rank consistency: {ranks}")


def test_sb039a_dates_and_age():
    text = (
        "Discharge Summary\n"
        "Patient age 62 years, primary osteoarthritis bilateral knees.\n"
        "Date of admission: 12-03-2026\n"
        "Date of discharge: 18-03-2026\n"
        "Total knee replacement performed."
    )
    pages = [
        _p("ds.pdf", "discharge_summary", text,
           extracted={"age": 62, "doa": "12-03-2026", "dod": "18-03-2026"}),
    ]
    rows = _build_case_rows("SB039A", "C3", pages)
    rows = assign_ranks("SB039A", rows)
    ok, issues = validate_rows("SB039A", rows)
    assert ok, issues
    r = rows[0]
    assert r["doa"] == "12-03-2026", r["doa"]
    assert r["dod"] == "18-03-2026", r["dod"]
    assert r["age_valid"] == 1, "age 62 > 55 should be valid"
    assert r["arthritis_type"] == 1, "osteoarthritis term should match"
    assert r["document_rank"] == 7
    print("[OK] SB039A dates and age")


def test_mg006a_dates_only_on_investigation_pages():
    pages = [
        _p("pre_inv.pdf", "investigation_pre",
           "Widal test 06/02/26\nO antigen positive\nFever 38.5",
           extracted={"dates_found": ["06/02/26"]}),
        _p("post_inv.pdf", "investigation_post",
           "Repeat Widal 12/02/26\nO antigen falling",
           extracted={"dates_found": ["12/02/26"]}),
        # Discharge summary should NOT fill pre_date / post_date.
        _p("ds.pdf", "discharge_summary",
           "Discharged on 14/02/26 in stable condition.",
           extracted={"dates_found": ["14/02/26"]}),
    ]
    rows = _build_case_rows("MG006A", "C4", pages)
    rows = assign_ranks("MG006A", rows)
    LK = "S3_link"  # MG006A's link field per HI.txt §C
    pre_row = next(r for r in rows if r[LK] == "pre_inv.pdf")
    post_row = next(r for r in rows if r[LK] == "post_inv.pdf")
    ds_row = next(r for r in rows if r[LK] == "ds.pdf")
    assert pre_row["pre_date"] == "06-02-2026", pre_row["pre_date"]
    assert post_row["post_date"] == "12-02-2026", post_row["post_date"]
    assert ds_row["pre_date"] is None and ds_row["post_date"] is None
    print("[OK] MG006A dates scoped to correct page types")


def main():
    test_mg064a_severe_anemia_case()
    test_multipage_pdf_consistency()
    test_sb039a_dates_and_age()
    test_mg006a_dates_only_on_investigation_pages()
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
