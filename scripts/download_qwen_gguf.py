"""Download Qwen3-VL-8B-Instruct GGUF + mmproj into ./models/.

The GGUF release from Qwen ships both the LM weights and the multimodal
projector. We pull whichever quantization is configured in nha_ps1/config.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from huggingface_hub import hf_hub_download

from nha_ps1.config import (
    MODELS_ROOT,
    QWEN_GGUF_FILE,
    QWEN_GGUF_REPO,
    QWEN_MMPROJ_FILE,
)


def main() -> int:
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)

    targets = [QWEN_GGUF_FILE, QWEN_MMPROJ_FILE]
    for fname in targets:
        target = MODELS_ROOT / fname
        if target.exists() and target.stat().st_size > 0:
            print(f"[skip] {target} already present")
            continue
        print(f"[download] {QWEN_GGUF_REPO} :: {fname} -> {MODELS_ROOT}")
        path = hf_hub_download(
            repo_id=QWEN_GGUF_REPO,
            filename=fname,
            local_dir=str(MODELS_ROOT),
            local_dir_use_symlinks=False,
        )
        print(f"[ok] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
