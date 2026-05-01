"""Centralized paths and runtime configuration."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("NHA_DATA_ROOT", PROJECT_ROOT / "Dataset-1" / "Claims"))
OUTPUT_ROOT = Path(os.environ.get("NHA_OUTPUT_ROOT", PROJECT_ROOT / "outputs"))
SYNTHETIC_LABELS_ROOT = Path(
    os.environ.get("NHA_SYNTHETIC_LABELS", PROJECT_ROOT / "data" / "synthetic_labels")
)
PROMPTS_ROOT = PROJECT_ROOT / "data" / "prompts"
MODELS_ROOT = Path(os.environ.get("NHA_MODELS_ROOT", PROJECT_ROOT / "models"))

PACKAGE_CODES = ["MG064A", "SG039C", "MG006A", "SB039A"]

PACKAGE_NAMES = {
    "MG064A": "Severe Anemia",
    "SG039C": "Cholecystectomy",
    "MG006A": "Enteric Fever",
    "SB039A": "Total Knee Replacement",
}

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}

# VLM backend selection
#   "llama_cpp"     -> Qwen3-VL-8B GGUF on CPU (default for final inference)
#   "transformers"  -> fp16/bf16 on Colab Pro GPU (used by colab_synthetic_label.py)
#   "cache_only"    -> reads from data/synthetic_labels/, errors if missing
VLM_BACKEND = os.environ.get("NHA_VLM_BACKEND", "llama_cpp")

QWEN_GGUF_REPO = "Qwen/Qwen3-VL-8B-Instruct-GGUF"
QWEN_GGUF_FILE = os.environ.get("NHA_QWEN_GGUF_FILE", "Qwen3-VL-8B-Instruct-Q4_K_M.gguf")
QWEN_MMPROJ_FILE = os.environ.get("NHA_QWEN_MMPROJ", "mmproj-Qwen3-VL-8B-Instruct-Q8_0.gguf")
QWEN_HF_REPO = "Qwen/Qwen3-VL-8B-Instruct"

# Inference render zoom for PDF→image rasterization (full(1).py used 2.0).
# 1.5 is enough for OCR while keeping CPU latency reasonable.
PDF_RENDER_ZOOM = float(os.environ.get("NHA_PDF_ZOOM", "1.5"))

# Page-quality blur threshold (Laplacian variance). Below this -> poor_quality=1.
BLUR_THRESHOLD = 80.0
