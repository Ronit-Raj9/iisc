"""Strict schema validation: every row's keys must match PACKAGE_SCHEMAS exactly.

HI.txt: 'Solutions that do not strictly adhere to the specified output format
will be rejected without evaluation.'
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .schemas import PACKAGE_SCHEMAS


def validate_rows(package_code: str, rows: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    expected = PACKAGE_SCHEMAS[package_code]
    issues: List[str] = []
    for i, row in enumerate(rows):
        actual = list(row.keys())
        if actual != expected:
            missing = [k for k in expected if k not in row]
            extra = [k for k in actual if k not in expected]
            issues.append(
                f"Row {i} ({package_code}): keys mismatch. "
                f"missing={missing} extra={extra} order_ok={actual == expected}"
            )
    return (len(issues) == 0, issues)


def validate_file(json_path: Path) -> Tuple[bool, List[str]]:
    package_code = json_path.stem  # e.g. MG064A.json -> MG064A
    if package_code not in PACKAGE_SCHEMAS:
        return (False, [f"Unknown package code: {package_code}"])
    with json_path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        return (False, [f"{json_path.name}: expected JSON array, got {type(rows).__name__}"])
    return validate_rows(package_code, rows)


def validate_dir(out_dir: Path) -> Dict[str, Tuple[bool, List[str]]]:
    results: Dict[str, Tuple[bool, List[str]]] = {}
    for code in PACKAGE_SCHEMAS:
        path = out_dir / f"{code}.json"
        if not path.exists():
            results[code] = (False, [f"Missing output file: {path}"])
            continue
        results[code] = validate_file(path)
    return results
