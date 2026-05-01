"""Walk the Dataset-1/Claims/<PACKAGE>/<case_id>/* tree."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from .config import SUPPORTED_EXTENSIONS, PACKAGE_CODES


def iter_case_files(case_dir: Path) -> List[Path]:
    """Return all supported document files under a case directory, sorted."""
    files: List[Path] = []
    for item in case_dir.rglob("*"):
        if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(item)
    return sorted(files)


def discover_cases(data_root: Path) -> Dict[str, Dict[str, List[Path]]]:
    """Group cases by package code.

    Returns: {package_code: {case_id: [Path, ...]}}
    """
    cases: Dict[str, Dict[str, List[Path]]] = {p: {} for p in PACKAGE_CODES}
    if not data_root.exists():
        return cases

    for package_dir in data_root.iterdir():
        if not package_dir.is_dir():
            continue
        package = package_dir.name
        if package not in cases:
            continue
        for case_dir in package_dir.iterdir():
            if case_dir.is_dir():
                files = iter_case_files(case_dir)
                if files:
                    cases[package][case_dir.name] = files
    return cases


def flatten_cases(cases: Dict[str, Dict[str, List[Path]]]) -> List[Tuple[str, str, List[Path]]]:
    """Flatten the nested case mapping into [(package, case_id, files), ...]."""
    out: List[Tuple[str, str, List[Path]]] = []
    for package, case_map in cases.items():
        for case_id, files in case_map.items():
            out.append((package, case_id, files))
    return out
