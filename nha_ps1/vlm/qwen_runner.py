"""Qwen3-VL-8B page analyzer with caching + multiple backends.

Backends:
- "cache_only":   read pre-computed JSON from data/synthetic_labels/.
- "llama_cpp":    Qwen3-VL-8B-Instruct-Q4_K_M.gguf via llama-cpp-python (CPU).
- "transformers": Qwen3-VL-8B-Instruct fp16/bf16 (Colab Pro / GPU).

Cache layout:
    data/synthetic_labels/<case_id>/<file_name>__page<N>.json

Always tries the cache first. The CPU path is the slow path; we want to hit
it as rarely as possible in the final run.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

try:
    from json_repair import repair_json  # type: ignore
    _HAS_REPAIR = True
except Exception:
    _HAS_REPAIR = False

from ..config import (
    QWEN_GGUF_FILE,
    QWEN_GGUF_REPO,
    QWEN_HF_REPO,
    QWEN_MMPROJ_FILE,
    SYNTHETIC_LABELS_ROOT,
    VLM_BACKEND,
)
from .prompts import (
    DOC_TYPE_VOCAB,
    PageAnalysis,
    SYSTEM_PROMPT,
    build_user_prompt,
)


# ---------- cache layer -------------------------------------------------------


def _cache_path(case_id: str, file_name: str, page_number: int) -> Path:
    safe = file_name.replace("/", "_").replace("\\", "_")
    return SYNTHETIC_LABELS_ROOT / case_id / f"{safe}__page{page_number}.json"


def load_cached(case_id: str, file_name: str, page_number: int) -> Optional[Dict[str, Any]]:
    path = _cache_path(case_id, file_name, page_number)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[cache] failed to read {path.name}: {e}")
        return None


def save_cached(case_id: str, file_name: str, page_number: int, payload: Dict[str, Any]) -> None:
    path = _cache_path(case_id, file_name, page_number)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[cache] failed to write {path.name}: {e}")


# ---------- JSON post-processing ---------------------------------------------


def _safe_json_loads(text: str) -> Dict[str, Any]:
    """Try strict JSON parse, then json-repair, then a brace-trim fallback."""
    text = text.strip()
    # Strip code fences if the model added them.
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    try:
        return json.loads(text)
    except Exception:
        pass
    if _HAS_REPAIR:
        try:
            return json.loads(repair_json(text))
        except Exception:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            pass
    return {}


def _normalize_payload(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce arbitrary VLM output into the PageAnalysis schema shape."""
    if not isinstance(raw, dict):
        raw = {}
    # Validate doc_type against the closed vocab.
    dt = str(raw.get("doc_type", "extra_document")).strip().lower()
    if dt not in DOC_TYPE_VOCAB:
        dt = "extra_document"
    raw["doc_type"] = dt
    try:
        return PageAnalysis(**raw).model_dump()
    except Exception:
        # If validation fails, return a minimal valid stub.
        stub = PageAnalysis(
            ocr_text=str(raw.get("ocr_text", "")),
            ocr_text_en=str(raw.get("ocr_text_en", raw.get("ocr_text", ""))),
            doc_type=dt,
            doc_type_confidence=float(raw.get("doc_type_confidence", 0.0) or 0.0),
        )
        return stub.model_dump()


# ---------- backends ----------------------------------------------------------


_LLAMA_INSTANCE = None
_TRANSFORMERS_INSTANCE = None


def _image_to_data_uri(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    b = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b}"


def _get_llama_cpp():
    """Lazily build a Llama (multimodal) handle. Returns None on import failure."""
    global _LLAMA_INSTANCE
    if _LLAMA_INSTANCE is not None:
        return _LLAMA_INSTANCE
    try:
        from llama_cpp import Llama  # type: ignore
        from llama_cpp.llama_chat_format import Qwen25VLChatHandler  # type: ignore
    except Exception as e:
        print(f"[qwen] llama-cpp-python not available: {e}")
        return None

    # Resolve model paths: env override first, then ./models/, then HF cache.
    from ..config import MODELS_ROOT

    gguf = MODELS_ROOT / QWEN_GGUF_FILE
    mmproj = MODELS_ROOT / QWEN_MMPROJ_FILE

    if not gguf.exists() or not mmproj.exists():
        print(
            f"[qwen] model files missing under {MODELS_ROOT}.\n"
            "       Run: python scripts/download_qwen_gguf.py"
        )
        return None

    try:
        chat_handler = Qwen25VLChatHandler(clip_model_path=str(mmproj), verbose=False)
        _LLAMA_INSTANCE = Llama(
            model_path=str(gguf),
            chat_handler=chat_handler,
            n_ctx=4096,
            n_threads=int(os.environ.get("NHA_LLAMA_THREADS", "8")),
            n_gpu_layers=int(os.environ.get("NHA_LLAMA_GPU_LAYERS", "0")),
            verbose=False,
        )
    except Exception as e:
        print(f"[qwen] failed to init llama.cpp: {e}")
        return None
    return _LLAMA_INSTANCE


def _llama_cpp_call(image: Image.Image, package_code: str) -> Dict[str, Any]:
    llm = _get_llama_cpp()
    if llm is None:
        return {}
    data_uri = _image_to_data_uri(image)
    user_prompt = build_user_prompt(package_code)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    try:
        resp = llm.create_chat_completion(
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=2048,
        )
        text = resp["choices"][0]["message"]["content"]
        return _safe_json_loads(text)
    except Exception as e:
        print(f"[qwen] llama_cpp inference error: {e}")
        return {}


def _get_transformers():
    """Lazy fp16 transformers backend for Colab Pro."""
    global _TRANSFORMERS_INSTANCE
    if _TRANSFORMERS_INSTANCE is not None:
        return _TRANSFORMERS_INSTANCE
    try:
        import torch  # type: ignore
        from transformers import (  # type: ignore
            AutoModelForVision2Seq,
            AutoProcessor,
        )
    except Exception as e:
        print(f"[qwen] transformers/torch unavailable: {e}")
        return None

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    try:
        processor = AutoProcessor.from_pretrained(QWEN_HF_REPO, trust_remote_code=True)
        model = AutoModelForVision2Seq.from_pretrained(
            QWEN_HF_REPO,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()
        _TRANSFORMERS_INSTANCE = (model, processor)
    except Exception as e:
        print(f"[qwen] failed to load HF model: {e}")
        return None
    return _TRANSFORMERS_INSTANCE


def _transformers_call(image: Image.Image, package_code: str) -> Dict[str, Any]:
    pair = _get_transformers()
    if pair is None:
        return {}
    model, processor = pair
    import torch  # type: ignore

    user_prompt = build_user_prompt(package_code)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    try:
        text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
            )
        generated = output_ids[:, inputs["input_ids"].shape[1] :]
        decoded = processor.batch_decode(generated, skip_special_tokens=True)[0]
        return _safe_json_loads(decoded)
    except Exception as e:
        print(f"[qwen] transformers inference error: {e}")
        return {}


# ---------- public entry point ------------------------------------------------


def analyze_page(
    image: Image.Image,
    package_code: str,
    case_id: str,
    file_name: str,
    page_number: int,
    backend: Optional[str] = None,
    use_cache: bool = True,
) -> Dict[str, Any]:
    """Run one VLM call (or hit cache) and return a PageAnalysis-shaped dict."""
    if use_cache:
        cached = load_cached(case_id, file_name, page_number)
        if cached:
            return _normalize_payload(cached)

    chosen = (backend or VLM_BACKEND).lower()
    if chosen == "cache_only":
        # Fail-soft: empty analysis so downstream populates a conservative row.
        return _normalize_payload({})

    if chosen == "transformers":
        raw = _transformers_call(image, package_code)
    else:  # default: llama_cpp
        raw = _llama_cpp_call(image, package_code)

    payload = _normalize_payload(raw)
    if use_cache and payload.get("ocr_text"):
        save_cached(case_id, file_name, page_number, payload)
    return payload
