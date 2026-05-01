"""End-to-end pipeline: discover cases -> per-page VLM -> rules -> ranks -> JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from .classify import resolve_doc_type
from .config import DATA_ROOT, OUTPUT_ROOT, PACKAGE_CODES
from .ingest import discover_cases
from .pages import estimate_page_quality, extract_pages
from .populate import populate_row
from .rank import assign_ranks
from .schemas import PACKAGE_SCHEMAS
from .validate import validate_rows
from .visual import detect_codes, merge_visual_signals
from .vlm.qwen_runner import analyze_page


def _process_page(
    package_code: str,
    case_id: str,
    file_name: str,
    page_number: int,
    image,
    backend: Optional[str],
) -> Dict[str, Any]:
    vlm = analyze_page(
        image=image,
        package_code=package_code,
        case_id=case_id,
        file_name=file_name,
        page_number=page_number,
        backend=backend,
    )
    text = vlm.get("ocr_text", "") or ""
    text_en = vlm.get("ocr_text_en", "") or text

    quality = estimate_page_quality(image, text)
    if vlm.get("page_quality"):
        # If the VLM also reported quality, OR them.
        vlm_q = vlm["page_quality"]
        quality["is_blurry"] = bool(quality.get("is_blurry") or vlm_q.get("is_blurry"))
        quality["is_poor_quality"] = bool(
            quality.get("is_poor_quality") or vlm_q.get("is_blurry")
        )

    code_visual = detect_codes(image)
    visual_tags = merge_visual_signals(vlm.get("visual_elements") or {}, code_visual)

    doc_type, doc_conf = resolve_doc_type(vlm, visual_tags)

    return {
        "doc_type": doc_type,
        "doc_type_confidence": doc_conf,
        "visual_tags": visual_tags,
        "quality": quality,
        "text": text,
        "text_en": text_en,
        "extracted": vlm.get("extracted") or {},
    }


def process_case(
    package_code: str,
    case_id: str,
    file_paths: List[Path],
    backend: Optional[str] = None,
    progress: bool = True,
) -> List[Dict[str, Any]]:
    """Process every page of every file in a case; return PACKAGE_SCHEMAS rows."""
    page_records: List[Dict[str, Any]] = []
    pbar = tqdm(file_paths, desc=f"{package_code}/{case_id}", disable=not progress, leave=False)
    for fp in pbar:
        for page in extract_pages(fp):
            rec = _process_page(
                package_code=package_code,
                case_id=case_id,
                file_name=fp.name,
                page_number=page["page_number"],
                image=page["image"],
                backend=backend,
            )
            rec["file_name"] = fp.name
            rec["page_number"] = page["page_number"]
            page_records.append(rec)

    # Build case-level joined text so package rules can look across pages
    # (e.g., severe_anemia phrase may live on a different page than the Hb lab).
    case_text = "\n".join(r["text"] for r in page_records if r.get("text"))
    case_text_en = "\n".join(r["text_en"] for r in page_records if r.get("text_en"))

    rows: List[Dict[str, Any]] = []
    for rec in page_records:
        row = populate_row(
            package_code=package_code,
            case_id=case_id,
            file_name=rec["file_name"],
            page_number=rec["page_number"],
            doc_type=rec["doc_type"],
            visual_tags=rec["visual_tags"],
            quality=rec["quality"],
            text=rec["text"],
            text_en=rec["text_en"],
            extracted=rec["extracted"],
            case_text=case_text,
            case_text_en=case_text_en,
        )
        rows.append(row)

    rows = assign_ranks(package_code, rows)
    return rows


def run_batch(
    data_root: Path = DATA_ROOT,
    output_root: Path = OUTPUT_ROOT,
    packages: Optional[List[str]] = None,
    case_limit: Optional[int] = None,
    backend: Optional[str] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """Run the pipeline for selected packages and write outputs/<PACKAGE>.json."""
    output_root.mkdir(parents=True, exist_ok=True)
    cases = discover_cases(data_root)
    targets = packages or PACKAGE_CODES

    all_results: Dict[str, List[Dict[str, Any]]] = {}
    for code in targets:
        case_map = cases.get(code, {})
        items = list(case_map.items())
        if case_limit:
            items = items[:case_limit]
        rows: List[Dict[str, Any]] = []
        print(f"\n=== {code} ({len(items)} cases) ===")
        for case_id, files in tqdm(items, desc=code):
            case_rows = process_case(code, case_id, files, backend=backend, progress=False)
            rows.extend(case_rows)

        ok, issues = validate_rows(code, rows)
        if not ok:
            print(f"[validate] {code} has issues:")
            for i in issues[:3]:
                print(f"  - {i}")

        out_path = output_root / f"{code}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        print(f"[write] {out_path} ({len(rows)} rows)")
        all_results[code] = rows
    return all_results
