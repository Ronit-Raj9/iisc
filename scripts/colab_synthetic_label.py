"""Run on Colab Pro (Tesla T4 / L4) to auto-label every page in Dataset-1.

Usage on Colab:
    !pip install -q transformers accelerate pymupdf pillow pydantic json-repair tqdm
    # upload Dataset-1/ to /content/
    !python scripts/colab_synthetic_label.py \
        --data-root /content/Dataset-1/Claims \
        --out-root /content/synthetic_labels

After the run completes, zip and download `synthetic_labels/` and copy it to
`data/synthetic_labels/` in the project; the CPU pipeline will read from there
and skip live VLM calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the parent package importable when run from the project root on Colab.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tqdm import tqdm

from nha_ps1.config import PACKAGE_CODES
from nha_ps1.ingest import discover_cases
from nha_ps1.pages import extract_pages
from nha_ps1.vlm.qwen_runner import analyze_page


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True, help="Path to Dataset-1/Claims/")
    p.add_argument("--out-root", required=True, help="Output dir for cached labels.")
    p.add_argument("--package", choices=PACKAGE_CODES, default=None)
    p.add_argument("--limit", type=int, default=None, help="Max cases per package.")
    args = p.parse_args()

    # Force the transformers backend AND override the cache root.
    import os
    os.environ["NHA_VLM_BACKEND"] = "transformers"
    os.environ["NHA_SYNTHETIC_LABELS"] = args.out_root

    # Re-import after env override.
    import importlib
    from nha_ps1 import config as _cfg
    importlib.reload(_cfg)
    from nha_ps1.vlm import qwen_runner as _qr
    importlib.reload(_qr)

    cases = discover_cases(Path(args.data_root))
    targets = [args.package] if args.package else PACKAGE_CODES

    for code in targets:
        case_map = cases.get(code, {})
        items = list(case_map.items())
        if args.limit:
            items = items[: args.limit]
        for case_id, files in tqdm(items, desc=code):
            for fp in files:
                for page in extract_pages(fp):
                    _qr.analyze_page(
                        image=page["image"],
                        package_code=code,
                        case_id=case_id,
                        file_name=fp.name,
                        page_number=page["page_number"],
                        backend="transformers",
                    )
    print("Done. Cache written to", args.out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
