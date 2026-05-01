"""Command-line entry point.

Usage:
    python -m nha_ps1 run --all
    python -m nha_ps1 run --package MG064A --limit 1
    python -m nha_ps1 validate outputs/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import DATA_ROOT, OUTPUT_ROOT, PACKAGE_CODES
from .pipeline import run_batch
from .validate import validate_dir


def _cmd_run(args: argparse.Namespace) -> int:
    packages = PACKAGE_CODES if args.all else [args.package]
    if args.package and args.package not in PACKAGE_CODES:
        print(f"unknown package: {args.package}", file=sys.stderr)
        return 2
    run_batch(
        data_root=Path(args.data_root),
        output_root=Path(args.output_root),
        packages=packages,
        case_limit=args.limit,
        backend=args.backend,
    )
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    out_dir = Path(args.path)
    results = validate_dir(out_dir)
    failed = 0
    for code, (ok, issues) in results.items():
        status = "OK" if ok else "FAIL"
        print(f"{status}  {code}.json")
        if not ok:
            failed += 1
            for i in issues[:5]:
                print(f"  - {i}")
    return 1 if failed else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser("nha_ps1")
    sub = p.add_subparsers(dest="cmd", required=True)

    pr = sub.add_parser("run", help="Run the pipeline.")
    pr.add_argument("--package", choices=PACKAGE_CODES, default=None)
    pr.add_argument("--all", action="store_true", help="Process all four packages.")
    pr.add_argument("--limit", type=int, default=None, help="Max cases per package.")
    pr.add_argument("--data-root", default=str(DATA_ROOT))
    pr.add_argument("--output-root", default=str(OUTPUT_ROOT))
    pr.add_argument(
        "--backend",
        choices=["llama_cpp", "transformers", "cache_only"],
        default=None,
        help="Override NHA_VLM_BACKEND.",
    )
    pr.set_defaults(func=_cmd_run)

    pv = sub.add_parser("validate", help="Validate output JSON files.")
    pv.add_argument("path", help="Directory containing <PACKAGE>.json files.")
    pv.set_defaults(func=_cmd_validate)

    args = p.parse_args(argv)
    if args.cmd == "run" and not args.all and not args.package:
        print("specify --package CODE or --all", file=sys.stderr)
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
