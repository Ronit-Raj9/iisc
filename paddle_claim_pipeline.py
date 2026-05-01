from __future__ import annotations

import hashlib
import contextlib
import io
import json
import math
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps

try:
    # Some notebook images have a NumPy/OpenCV ABI mismatch that writes a noisy
    # AttributeError to stderr before import failure. Keep the pipeline usable
    # by treating OpenCV as optional and suppressing that import noise.
    with contextlib.redirect_stderr(io.StringIO()):
        import cv2  # type: ignore
except Exception:  # pragma: no cover - notebook fallback
    cv2 = None

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover - notebook fallback
    fitz = None

try:
    from rapidfuzz import fuzz as _rf_fuzz  # type: ignore
except Exception:  # pragma: no cover
    _rf_fuzz = None

NB: Dict[str, Any] = {}
_PADDLE_OCR = None
_ORIGINAL_ANALYZE_PAGE_WITH_GEMMA = None
_ORIGINAL_ANALYZE_CLINICAL_WITH_GEMMA = None
_ORIGINAL_ANALYZE_CLAIM_RECONCILE = None


SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def install_paddle_pipeline(globals_dict: Dict[str, Any]) -> None:
    """Install notebook overrides for the CPU/pip-only PaddleOCR pipeline."""
    global NB, _ORIGINAL_ANALYZE_PAGE_WITH_GEMMA, _ORIGINAL_ANALYZE_CLINICAL_WITH_GEMMA, _ORIGINAL_ANALYZE_CLAIM_RECONCILE
    NB = globals_dict
    _ORIGINAL_ANALYZE_PAGE_WITH_GEMMA = globals_dict.get("analyze_page_with_gemma")
    _ORIGINAL_ANALYZE_CLINICAL_WITH_GEMMA = globals_dict.get("analyze_clinical_with_gemma")
    _ORIGINAL_ANALYZE_CLAIM_RECONCILE = globals_dict.get("analyze_claim_reconcile_with_gemma")

    repo = Path.cwd()
    globals_dict.setdefault("OCR_CACHE_ROOT", Path("./ocr_cache"))
    globals_dict.setdefault("PAGE_CACHE_ROOT", Path("./page_cache"))
    globals_dict.setdefault("PADDLE_OCR_LANG", os.environ.get("PADDLE_OCR_LANG", "en"))
    globals_dict.setdefault("USE_PADDLE_OCR", os.environ.get("USE_PADDLE_OCR", "1") != "0")
    globals_dict.setdefault("USE_GEMMA_FALLBACK", os.environ.get("USE_GEMMA_FALLBACK", "1") != "0")
    globals_dict.setdefault("USE_GEMMA_FOR_ALL_PAGES", os.environ.get("USE_GEMMA_FOR_ALL_PAGES", "0") == "1")
    globals_dict.setdefault("USE_CLINICAL_GEMMA", os.environ.get("USE_CLINICAL_GEMMA", "0") == "1")
    globals_dict.setdefault("USE_CLAIM_LEVEL_GEMMA", os.environ.get("USE_CLAIM_LEVEL_GEMMA", "1") == "1")
    globals_dict.setdefault("USE_PAGE_IMAGE_CACHE", os.environ.get("USE_PAGE_IMAGE_CACHE", "1") == "1")
    globals_dict.setdefault("SKIP_FULL_OCR_ON_TEXT_PDF", os.environ.get("SKIP_FULL_OCR_ON_TEXT_PDF", "1") == "1")
    globals_dict.setdefault("MIN_GEMMA_DOC_CONF", float(os.environ.get("MIN_GEMMA_DOC_CONF", "0.72")))
    globals_dict.setdefault("EMBEDDED_TEXT_MIN_CHARS_SKIP_OCR", int(os.environ.get("EMBEDDED_TEXT_MIN_CHARS_SKIP_OCR", "400")))
    globals_dict.setdefault("DEDUPE_PAGES_BY_DHASH", os.environ.get("DEDUPE_PAGES_BY_DHASH", "1") == "1")
    globals_dict.setdefault("USE_OCR_ADAPTIVE_THRESHOLD", os.environ.get("USE_OCR_ADAPTIVE_THRESHOLD", "0") == "1")
    globals_dict["OCR_CACHE_ROOT"].mkdir(parents=True, exist_ok=True)
    globals_dict["PAGE_CACHE_ROOT"].mkdir(parents=True, exist_ok=True)

    globals_dict["build_dataset_manifest"] = build_dataset_manifest
    globals_dict["extract_pages"] = extract_pages
    globals_dict["preprocess_for_ocr"] = preprocess_for_ocr
    globals_dict["run_ocr"] = run_ocr
    globals_dict["filename_doc_type_hint"] = filename_doc_type_hint
    globals_dict["classify_document_type"] = classify_document_type
    globals_dict["detect_visual_elements"] = detect_visual_elements
    globals_dict["extract_deterministic_entities"] = extract_deterministic_entities
    globals_dict["reconstruct_lines_from_ocr"] = reconstruct_lines_from_ocr
    globals_dict["difference_hash16"] = difference_hash16
    globals_dict["build_claim_evidence_bundle"] = build_claim_evidence_bundle
    globals_dict["dedupe_page_results_by_dhash"] = dedupe_page_results_by_dhash
    globals_dict["process_case"] = process_case
    globals_dict["run_rules_engine"] = run_rules_engine
    globals_dict["PADDLE_PIPELINE_INSTALLED"] = True

    print("[paddle-pipeline] installed")
    print(f"[paddle-pipeline] repo={repo}")
    print("[paddle-pipeline] OCR=PaddleOCR CPU, PDF renderer=PyMuPDF, barcode=zxing-cpp")
    print("[paddle-pipeline] Gemma=selective page + optional claim reconcile; rules=deterministic")


def _nb(name: str, default: Any = None) -> Any:
    return NB.get(name, default)


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _image_hash(image: Image.Image) -> str:
    small = image.convert("L").resize((64, 64))
    return hashlib.sha1(small.tobytes()).hexdigest()[:16]


def difference_hash16(image: Image.Image) -> str:
    """16-byte difference hash for near-duplicate page detection (perceptual-lite)."""
    g = np.array(ImageOps.grayscale(image.resize((9, 8), Image.Resampling.LANCZOS)), dtype=np.int16)
    bits: List[int] = []
    for y in range(8):
        for x in range(8):
            bits.append(1 if g[y, x] > g[y, x + 1] else 0)
    out = 0
    for b in bits:
        out = (out << 1) | b
    return f"{out:016x}"


def _file_render_cache_key(file_path: Path) -> str:
    try:
        st = file_path.stat()
        raw = f"{file_path.resolve()}|{st.st_mtime_ns}|{st.st_size}".encode("utf-8", errors="replace")
    except Exception:
        raw = str(file_path).encode("utf-8", errors="replace")
    return hashlib.sha1(raw).hexdigest()[:24]


def _page_cache_base(file_path: Path) -> Path:
    root = _nb("PAGE_CACHE_ROOT", Path("./page_cache"))
    return root / _file_render_cache_key(file_path)


def _load_page_meta(meta_path: Path) -> Dict[str, Any]:
    if not meta_path.exists():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_page_meta(meta_path: Path, payload: Dict[str, Any]) -> None:
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def reconstruct_lines_from_ocr(ocr_lines: Sequence[Any], y_tol: int = 14) -> str:
    """Cluster OCR boxes by y-center and join into pseudo-lines for table/lab regex."""
    rows: List[Tuple[float, float, str]] = []
    for line in ocr_lines or []:
        d = _ocr_line_to_dict(line)
        t = (d.get("text") or "").strip()
        bb = d.get("bbox")
        if not t or not bb or len(bb) < 4:
            continue
        yc = (float(bb[1]) + float(bb[3])) / 2.0
        xc = float(bb[0])
        rows.append((yc, xc, t))
    if not rows:
        return ""
    rows.sort(key=lambda r: (r[0], r[1]))
    lines_out: List[List[str]] = []
    current: List[Tuple[float, str]] = []
    last_y: Optional[float] = None
    for yc, xc, t in rows:
        if last_y is None or abs(yc - last_y) <= y_tol:
            current.append((xc, t))
        else:
            current.sort(key=lambda z: z[0])
            lines_out.append([x[1] for x in current])
            current = [(xc, t)]
        last_y = yc
    if current:
        current.sort(key=lambda z: z[0])
        lines_out.append([x[1] for x in current])
    return "\n".join(" ".join(parts) for parts in lines_out)


def _ocr_cache_path(case_id: str, file_name: str, page_number: int, image: Image.Image) -> Path:
    root = _nb("OCR_CACHE_ROOT", Path("./ocr_cache"))
    stem = f"{_safe_name(file_name)}__p{page_number}__{_image_hash(image)}.json"
    return root / _safe_name(case_id) / stem


def _make_ocr_line(text: str, bbox: Optional[List[int]], confidence: Optional[float]) -> Any:
    cls = _nb("OCRLine")
    if cls is None:
        return {"text": text, "bbox": bbox, "confidence": confidence}
    return cls(text=text, bbox=bbox, confidence=confidence)


def _ocr_line_to_dict(line: Any) -> Dict[str, Any]:
    if isinstance(line, dict):
        return line
    return {
        "text": getattr(line, "text", ""),
        "bbox": getattr(line, "bbox", None),
        "confidence": getattr(line, "confidence", None),
    }


def _ocr_line_from_dict(data: Dict[str, Any]) -> Any:
    return _make_ocr_line(data.get("text", ""), data.get("bbox"), data.get("confidence"))


def build_dataset_manifest(data_root: Path) -> List[Dict[str, Any]]:
    """Create a lightweight manifest for auditing and pipeline triage."""
    rows: List[Dict[str, Any]] = []
    supported = _nb("SUPPORTED_EXTENSIONS", {".pdf", *SUPPORTED_IMAGE_EXTS})
    for file_path in sorted(data_root.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in supported:
            continue
        parts = file_path.relative_to(data_root).parts
        package_code = parts[0] if len(parts) > 0 else None
        case_id = parts[1] if len(parts) > 1 else None
        row: Dict[str, Any] = {
            "package_code": package_code,
            "case_id": case_id,
            "file_name": file_path.name,
            "path": str(file_path),
            "extension": file_path.suffix.lower(),
            "size_bytes": file_path.stat().st_size,
            "page_count": 1,
            "embedded_text_chars": 0,
            "is_probably_scanned": None,
        }
        if file_path.suffix.lower() == ".pdf" and fitz is not None:
            try:
                doc = fitz.open(str(file_path))
                row["page_count"] = len(doc)
                text_chars = 0
                for page in doc:
                    text_chars += len((page.get_text("text") or "").strip())
                row["embedded_text_chars"] = text_chars
                row["is_probably_scanned"] = text_chars < max(80, 40 * len(doc))
                doc.close()
            except Exception as exc:
                row["pdf_error"] = str(exc)
        elif file_path.suffix.lower() in SUPPORTED_IMAGE_EXTS:
            row["is_probably_scanned"] = True
            try:
                with Image.open(file_path) as img:
                    row["image_width"] = img.width
                    row["image_height"] = img.height
            except Exception as exc:
                row["image_error"] = str(exc)
        rows.append(row)
    return rows


def extract_pages(file_path: Path) -> List[Dict[str, Any]]:
    """Render PDFs with PyMuPDF and wrap images as single-page records; optional PNG page cache."""
    ext = file_path.suffix.lower()
    pages: List[Dict[str, Any]] = []
    use_cache = bool(_nb("USE_PAGE_IMAGE_CACHE", True))
    cache_dir = _page_cache_base(file_path) if use_cache else None

    if ext == ".pdf":
        if fitz is None:
            print(f"[extract_pages] PyMuPDF unavailable; cannot render {file_path.name}")
            return pages
        try:
            doc = fitz.open(str(file_path))
            zoom = float(_nb("PDF_RENDER_ZOOM", 2.0))
            mat = fitz.Matrix(zoom, zoom)
            for idx, page in enumerate(doc):
                embedded_text = (page.get_text("text") or "").strip()
                pnum = idx + 1
                img: Optional[Image.Image] = None
                if cache_dir is not None:
                    png_path = cache_dir / f"p{pnum}.png"
                    meta_path = cache_dir / f"p{pnum}.json"
                    if png_path.exists() and meta_path.exists():
                        meta = _load_page_meta(meta_path)
                        try:
                            img = Image.open(png_path).convert("RGB")
                            embedded_text = str(meta.get("embedded_text", embedded_text))
                        except Exception:
                            img = None
                if img is None:
                    pix = page.get_pixmap(matrix=mat, alpha=False)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    if cache_dir is not None:
                        try:
                            cache_dir.mkdir(parents=True, exist_ok=True)
                            img.save(cache_dir / f"p{pnum}.png", format="PNG")
                            _save_page_meta(
                                cache_dir / f"p{pnum}.json",
                                {"embedded_text": embedded_text, "page_number": pnum, "file_name": file_path.name},
                            )
                        except Exception:
                            pass
                pages.append({
                    "page_number": pnum,
                    "image": img,
                    "file_name": file_path.name,
                    "source_path": str(file_path),
                    "embedded_text": embedded_text,
                    "embedded_text_chars": len(embedded_text),
                })
            doc.close()
        except Exception as exc:
            print(f"[extract_pages] PDF '{file_path.name}' failed: {exc}")
    elif ext in SUPPORTED_IMAGE_EXTS:
        try:
            pnum = 1
            img: Optional[Image.Image] = None
            if cache_dir is not None:
                png_path = cache_dir / f"p{pnum}.png"
                if png_path.exists():
                    try:
                        img = Image.open(png_path).convert("RGB")
                    except Exception:
                        img = None
            if img is None:
                img = Image.open(file_path)
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")
                if cache_dir is not None:
                    try:
                        cache_dir.mkdir(parents=True, exist_ok=True)
                        img.save(cache_dir / f"p{pnum}.png", format="PNG")
                        _save_page_meta(
                            cache_dir / f"p{pnum}.json",
                            {"embedded_text": "", "page_number": pnum, "file_name": file_path.name},
                        )
                    except Exception:
                        pass
            pages.append({
                "page_number": pnum,
                "image": img,
                "file_name": file_path.name,
                "source_path": str(file_path),
                "embedded_text": "",
                "embedded_text_chars": 0,
            })
        except Exception as exc:
            print(f"[extract_pages] image '{file_path.name}' failed: {exc}")
    return pages


def _is_xray_like(file_name: str = "", text: str = "") -> bool:
    lo = f"{file_name} {text}".lower()
    return any(token in lo for token in ["xray", "x-ray", "x ray", "ct knee", "post_xray", "knee ap", "knee lateral"])


def _cap_long_edge(image: Image.Image, max_edge: int = 2400, min_edge: int = 1200) -> Image.Image:
    w, h = image.size
    long_edge = max(w, h)
    if long_edge > max_edge:
        scale = max_edge / float(long_edge)
        return image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    if long_edge < min_edge:
        scale = min_edge / float(long_edge)
        return image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    return image


def _crop_border(gray: np.ndarray) -> np.ndarray:
    if cv2 is None or gray.size == 0:
        return gray
    try:
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Keep both dark text on light background and light xray labels on dark background.
        mask = thresh < 250
        ys, xs = np.where(mask)
        if len(xs) < 100 or len(ys) < 100:
            return gray
        x1, x2 = max(0, xs.min() - 8), min(gray.shape[1], xs.max() + 8)
        y1, y2 = max(0, ys.min() - 8), min(gray.shape[0], ys.max() + 8)
        if (x2 - x1) < gray.shape[1] * 0.45 or (y2 - y1) < gray.shape[0] * 0.45:
            return gray
        return gray[y1:y2, x1:x2]
    except Exception:
        return gray


def _deskew(gray: np.ndarray) -> np.ndarray:
    if cv2 is None or gray.size == 0:
        return gray
    try:
        inv = cv2.bitwise_not(gray)
        coords = np.column_stack(np.where(inv > 0))
        if len(coords) < 200:
            return gray
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) < 0.3 or abs(angle) > 10:
            return gray
        h, w = gray.shape[:2]
        matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        return cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        return gray


def preprocess_for_ocr(image: Image.Image, file_name: str = "", text_hint: str = "") -> Image.Image:
    """CPU-safe preprocessing. X-rays keep grayscale; documents get OCR contrast cleanup."""
    img = _cap_long_edge(image.convert("RGB"))
    gray = np.array(ImageOps.grayscale(img))
    xray = _is_xray_like(file_name, text_hint)
    if cv2 is None:
        return ImageOps.autocontrast(Image.fromarray(gray)).convert("RGB")

    gray = _crop_border(gray)
    if not xray:
        gray = _deskew(gray)
    try:
        if xray:
            clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
            out = clahe.apply(gray)
        else:
            denoised = cv2.fastNlMeansDenoising(gray, None, 8, 7, 21)
            clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
            out = clahe.apply(denoised)
            if bool(_nb("USE_OCR_ADAPTIVE_THRESHOLD", False)):
                try:
                    out = cv2.adaptiveThreshold(
                        out, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 11,
                    )
                except Exception:
                    pass
        return Image.fromarray(out).convert("RGB")
    except Exception:
        return ImageOps.autocontrast(Image.fromarray(gray)).convert("RGB")


def _get_paddle_ocr():
    global _PADDLE_OCR
    if _PADDLE_OCR is not None:
        return _PADDLE_OCR
    if not _nb("USE_PADDLE_OCR", True):
        return None
    try:
        import paddle  # type: ignore

        paddle.set_flags({"FLAGS_use_mkldnn": False})
    except Exception:
        pass
    try:
        from paddleocr import PaddleOCR  # type: ignore
    except Exception as exc:
        print(f"[ocr] PaddleOCR unavailable: {exc}")
        return None

    lang = _nb("PADDLE_OCR_LANG", "en")
    # PaddleOCR 2.x / 3.x differ: some wheels reject show_log, use_gpu, etc.
    init_attempts = [
        dict(use_angle_cls=True, lang=lang),
        dict(lang=lang),
        dict(),
    ]
    last_err: Optional[str] = None
    for kwargs in init_attempts:
        try:
            _PADDLE_OCR = PaddleOCR(**kwargs)
            return _PADDLE_OCR
        except Exception as exc:
            last_err = str(exc)
            print(f"[ocr] PaddleOCR init failed with {kwargs!r}: {exc}")
            continue
    print(f"[ocr] PaddleOCR: all init variants failed ({last_err})")
    return None


def _looks_like_bbox(value: Any) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return False
    if all(isinstance(p, (list, tuple)) and len(p) >= 2 for p in value[:4]):
        return True
    return all(isinstance(p, (int, float)) for p in value[:4])


def _bbox_to_xyxy(box: Any) -> Optional[List[int]]:
    try:
        if all(isinstance(p, (list, tuple)) and len(p) >= 2 for p in box[:4]):
            xs = [float(p[0]) for p in box[:4]]
            ys = [float(p[1]) for p in box[:4]]
            return [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]
        vals = [float(x) for x in box[:4]]
        return [int(vals[0]), int(vals[1]), int(vals[2]), int(vals[3])]
    except Exception:
        return None


def _parse_paddle_output(result: Any) -> List[Any]:
    lines: List[Any] = []

    def visit(node: Any) -> None:
        if node is None:
            return
        if isinstance(node, dict):
            texts = node.get("rec_texts") or node.get("texts") or []
            scores = node.get("rec_scores") or node.get("scores") or []
            boxes = node.get("dt_polys") or node.get("rec_polys") or node.get("boxes") or []
            for i, text in enumerate(texts):
                conf = float(scores[i]) if i < len(scores) else None
                bbox = _bbox_to_xyxy(boxes[i]) if i < len(boxes) else None
                if str(text).strip():
                    lines.append(_make_ocr_line(str(text).strip(), bbox, conf))
            return
        if isinstance(node, (list, tuple)):
            if len(node) >= 2 and _looks_like_bbox(node[0]) and isinstance(node[1], (list, tuple)):
                if node[1] and isinstance(node[1][0], str):
                    text = node[1][0].strip()
                    conf = None
                    if len(node[1]) > 1:
                        try:
                            conf = float(node[1][1])
                        except Exception:
                            conf = None
                    if text:
                        lines.append(_make_ocr_line(text, _bbox_to_xyxy(node[0]), conf))
                    return
            for child in node:
                visit(child)

    visit(result)
    lines.sort(key=lambda l: ((getattr(l, "bbox", None) or [0, 0, 0, 0])[1], (getattr(l, "bbox", None) or [0, 0, 0, 0])[0]))
    return lines


def _xray_corner_crops(page_image: Image.Image, frac: float = 0.24) -> List[Image.Image]:
    w, h = page_image.size
    fw, fh = max(32, int(w * frac)), max(32, int(h * frac))
    boxes = ((0, 0, fw, fh), (w - fw, 0, w, fh), (0, h - fh, fw, h), (w - fw, h - fh, w, h))
    out: List[Image.Image] = []
    for box in boxes:
        try:
            out.append(page_image.crop(box))
        except Exception:
            continue
    return out


def _paddle_lines_from_rgb(ocr: Any, rgb: Image.Image) -> List[Any]:
    arr = np.array(rgb.convert("RGB"))
    try:
        result = ocr.ocr(arr, cls=True)
    except TypeError:
        result = ocr.predict(arr) if hasattr(ocr, "predict") else ocr.ocr(arr)
    return _parse_paddle_output(result)


def _merge_ocr_line_lists(primary: List[Any], extra: List[Any]) -> List[Any]:
    seen = set()
    out: List[Any] = []
    for line in primary + extra:
        d = _ocr_line_to_dict(line)
        key = (d.get("text") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(line)
    return out


def run_ocr(
    page_image: Image.Image,
    case_id: Optional[str] = None,
    file_name: Optional[str] = None,
    page_number: Optional[int] = None,
    embedded_text: str = "",
) -> Tuple[str, List[Any]]:
    """Run cached PaddleOCR, optional skip for text-native PDFs, X-ray corner pass, merge embedded text."""
    file_name = file_name or "page"
    page_number = int(page_number or 1)
    case_id = case_id or "_adhoc"
    cache_path = _ocr_cache_path(case_id, file_name, page_number, page_image)
    if cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            lines = [_ocr_line_from_dict(x) for x in payload.get("lines", [])]
            return payload.get("text", ""), lines
        except Exception:
            pass

    embedded = (embedded_text or "").strip()
    min_skip = int(_nb("EMBEDDED_TEXT_MIN_CHARS_SKIP_OCR", 400))
    skip_full = bool(_nb("SKIP_FULL_OCR_ON_TEXT_PDF", True)) and len(embedded) >= min_skip
    xray = _is_xray_like(file_name, embedded)
    ocr = _get_paddle_ocr()
    lines: List[Any] = []
    ocr_text = ""

    if ocr is not None and not skip_full:
        try:
            prep = preprocess_for_ocr(page_image, file_name=file_name, text_hint=embedded)
            lines = _paddle_lines_from_rgb(ocr, prep)
            ocr_text = "\n".join(_ocr_line_to_dict(line)["text"] for line in lines if _ocr_line_to_dict(line).get("text"))
        except Exception as exc:
            print(f"[ocr] {case_id}/{file_name} p{page_number} failed: {exc}")
    elif skip_full and ocr is not None and xray:
        # Labels on film: light OCR on corners only when skipping full page.
        try:
            corner_lines: List[Any] = []
            for crop in _xray_corner_crops(page_image):
                arr = np.array(ImageOps.autocontrast(crop.convert("RGB")))
                try:
                    result = ocr.ocr(arr, cls=True)
                except TypeError:
                    result = ocr.predict(arr) if hasattr(ocr, "predict") else ocr.ocr(arr)
                corner_lines.extend(_parse_paddle_output(result))
            lines = corner_lines
            ocr_text = "\n".join(_ocr_line_to_dict(line)["text"] for line in lines if _ocr_line_to_dict(line).get("text"))
        except Exception as exc:
            print(f"[ocr-corner] {case_id}/{file_name} p{page_number}: {exc}")
    elif skip_full:
        lines = []
        ocr_text = ""

    if ocr is not None and xray and not skip_full:
        try:
            extras: List[Any] = []
            for crop in _xray_corner_crops(page_image):
                prep_c = ImageOps.autocontrast(crop.convert("RGB"))
                extras.extend(_paddle_lines_from_rgb(ocr, prep_c))
            lines = _merge_ocr_line_lists(lines, extras)
            ocr_text = "\n".join(_ocr_line_to_dict(line)["text"] for line in lines if _ocr_line_to_dict(line).get("text"))
        except Exception:
            pass

    if embedded and len(embedded) > 80:
        text = embedded if len(embedded) >= len(ocr_text) else f"{embedded}\n{ocr_text}"
    else:
        text = ocr_text or embedded

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({
        "text": text,
        "embedded_text_chars": len(embedded),
        "ocr_text_chars": len(ocr_text),
        "skipped_full_ocr": int(bool(skip_full)),
        "lines": [_ocr_line_to_dict(line) for line in lines],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return text, lines


DATE_PATTERNS = [
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    r"\b\d{1,2}[.]\d{1,2}[.]\d{2,4}\b",
    r"\b\d{1,2}-[A-Za-z]{3}-\d{2,4}\b",
    r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}\b",
    r"\b[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4}\b",
]


def _find_dates(text: str) -> List[str]:
    seen, out = set(), []
    for pat in DATE_PATTERNS:
        for match in re.findall(pat, text or "", flags=re.IGNORECASE):
            if match not in seen:
                seen.add(match)
                out.append(match)
    return out


def _find_label_date(text: str, labels: Sequence[str]) -> Optional[str]:
    if not text:
        return None
    for label in labels:
        pat = rf"{re.escape(label)}\s*[:=\-]?\s*([0-9]{{1,2}}[./-][0-9]{{1,2}}[./-][0-9]{{2,4}})"
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def _normalize_date(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    notebook_normalize = _nb("normalize_date")
    if callable(notebook_normalize):
        try:
            return notebook_normalize(value)
        except Exception:
            pass
    s = str(value).strip()
    for fmt in ("%d/%m/%y", "%d/%m/%Y", "%d-%m-%y", "%d-%m-%Y", "%d.%m.%y", "%d.%m.%Y", "%d-%b-%y", "%d-%b-%Y"):
        try:
            dt = datetime.strptime(s, fmt)
            if dt.year < 1950:
                dt = dt.replace(year=dt.year + 100)
            return dt.strftime("%d-%m-%Y")
        except Exception:
            continue
    return s


def _find_age(text: str) -> Optional[int]:
    patterns = [
        r"\bage\s*[:=\-]?\s*(\d{1,3})\b",
        r"\b(\d{1,3})\s*(?:years?|yrs?|y/o|yo|year old)\b",
        r"\b(?:m|f)\s*/\s*(\d{1,3})\b",
        r"\b(\d{1,3})\s*/\s*(?:m|f)\b",
    ]
    for pat in patterns:
        m = re.search(pat, text or "", flags=re.IGNORECASE)
        if m:
            age = int(m.group(1))
            if 0 < age < 120:
                return age
    return None


def _hb_values(text: str) -> List[float]:
    values: List[float] = []
    patterns = [
        r"\b(?:hb|hgb|h\.b\.|haemoglobin|hemoglobin)\s*[:=\-]?\s*(\d{1,2}(?:\.\d{1,2})?)\b",
        r"\b(\d{1,2}(?:\.\d{1,2})?)\s*(?:g/dl|gm/dl|g/dL)\b.{0,30}\b(?:hb|hgb|haemoglobin|hemoglobin)\b",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text or "", flags=re.IGNORECASE):
            try:
                val = float(m.group(1))
                if 2.0 <= val <= 20.0:
                    values.append(val)
            except Exception:
                continue
    dedup: List[float] = []
    for v in values:
        if v not in dedup:
            dedup.append(v)
    return dedup


def _temperature_and_duration(text: str) -> Tuple[Optional[float], Optional[int]]:
    temp_c: Optional[float] = None
    for m in re.finditer(r"\b(?:temp(?:erature)?\s*[:=\-]?\s*)?(\d{2,3}(?:\.\d+)?)\s*°?\s*([CF])?\b", text or "", flags=re.IGNORECASE):
        val = float(m.group(1))
        unit = (m.group(2) or "").upper()
        if unit == "F" or val > 50:
            c = (val - 32) * 5.0 / 9.0
        else:
            c = val
        if 35.0 <= c <= 43.5:
            temp_c = max(temp_c or c, c)
    duration: Optional[int] = None
    for pat in [r"\bfever\s*(?:for|since)?\s*(\d{1,2})\s*days?\b", r"\b(\d{1,2})\s*days?\s*(?:of)?\s*fever\b", r"\bsince\s*(\d{1,2})\s*days?\b"]:
        m = re.search(pat, text or "", flags=re.IGNORECASE)
        if m:
            val = int(m.group(1))
            if 0 < val < 60:
                duration = max(duration or val, val)
    return temp_c, duration


SYMPTOM_TERMS = [
    "pallor", "pale", "fatigue", "weakness", "dizziness", "dyspnea", "tachycardia",
    "palpitations", "shock", "bleeding", "fever", "headache", "body ache",
    "vomiting", "diarrhea", "constipation", "right hypochondrium", "epigastrium",
    "abdominal pain", "ruq pain", "pain abdomen", "unable to walk", "severe pain",
]

DIAGNOSIS_TERMS = [
    "severe anemia", "severe anaemia", "enteric fever", "typhoid", "cholelithiasis",
    "cholecystitis", "gall stone", "gallstone", "osteoarthritis", "primary oa",
    "rheumatoid arthritis", "genu varum", "joint space narrowing",
]

TREATMENT_TERMS = [
    "blood transfusion", "blood transfused", "packed red blood cells", "prbc",
    "whole blood", "ferrous sulphate", "ferrous sulfate", "iron injection",
    "iv iron", "parenteral iron", "laparoscopic cholecystectomy", "total knee replacement",
]


def _terms_present(text: str, terms: Sequence[str]) -> List[str]:
    lo = (text or "").lower()
    return [term for term in terms if term in lo]


def _first_float_match(text: str, pattern: str) -> Optional[float]:
    m = re.search(pattern, text or "", flags=re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _extended_laboratory_fields(blob: str) -> Dict[str, Any]:
    """CBC / ESR / smear / LFT / laterality / follow-up from OCR+reconstructed text."""
    t = blob or ""
    wbc = _first_float_match(t, r"\b(?:wbc|tlc|total\s*leucocyte\s*count)\s*[:=\-]?\s*(\d+(?:\.\d+)?)\b")
    rbc = _first_float_match(t, r"\b(?:rbc|red\s*blood)\s*[:=\-]?\s*(\d+(?:\.\d+)?)\b")
    plt = _first_float_match(t, r"\b(?:platelet|plt)\s*[:=\-]?\s*(\d+(?:\.\d+)?)\b")
    hct = _first_float_match(t, r"\b(?:hct|hematocrit|packed\s*cell\s*volume|pcv)\s*[:=\-]?\s*(\d+(?:\.\d+)?)\b")
    esr = _first_float_match(t, r"\b(?:esr|erythrocyte\s*sedimentation)\s*[:=\-]?\s*(\d+(?:\.\d+)?)\b")
    bili = _first_float_match(t, r"\b(?:bilirubin|total\s*bilirubin|tbil)\s*[:=\-]?\s*(\d+(?:\.\d+)?)\b")
    ast = _first_float_match(t, r"\b(?:sgot|ast)\s*[:=\-]?\s*(\d+(?:\.\d+)?)\b")
    alt = _first_float_match(t, r"\b(?:sgpt|alt)\s*[:=\-]?\s*(\d+(?:\.\d+)?)\b")
    alp = _first_float_match(t, r"\b(?:alkaline\s*phosphatase|alp)\s*[:=\-]?\s*(\d+(?:\.\d+)?)\b")
    smear_hits: List[str] = []
    lo = t.lower()
    if "peripheral smear" in lo or re.search(r"\bpbs\b", lo):
        for term in ("normocytic", "microcytic", "macrocytic", "toxic", "malaria", "parasite", "dimorphic"):
            if term in lo:
                smear_hits.append(term)
    lat = None
    for token, lab in ((" left ", "left"), (" right ", "right"), (" lt ", "left"), (" rt ", "right")):
        if token in f" {lo} ":
            lat = lab
            break
    follow = _find_label_date(t, ["follow up", "follow-up", "review on", "next visit"])
    return {
        "wbc": wbc,
        "rbc": rbc,
        "platelet": plt,
        "hematocrit": hct,
        "esr_value": esr,
        "peripheral_smear_findings": smear_hits,
        "bilirubin_total": bili,
        "sgot_ast": ast,
        "sgpt_alt": alt,
        "alkaline_phosphatase": alp,
        "laterality": lat,
        "follow_up_date": follow,
    }


def extract_deterministic_entities(
    text: str,
    file_name: str = "",
    doc_type: str = "",
    ocr_lines: Optional[Sequence[Any]] = None,
) -> Dict[str, Any]:
    recon = reconstruct_lines_from_ocr(ocr_lines or [])
    blob = ((text or "") + "\n" + recon).strip()
    temp_c, fever_days = _temperature_and_duration(blob)
    dates = _find_dates(blob)
    labs = _extended_laboratory_fields(blob)
    out: Dict[str, Any] = {
        "patient_age": _find_age(blob),
        "dates_found": dates,
        "doa": _find_label_date(blob, ["date of admission", "admission date", "doa", "admitted on"]),
        "dod": _find_label_date(blob, ["date of discharge", "discharge date", "dod", "discharged on"]),
        "hb_values": _hb_values(blob),
        "temperature_celsius": temp_c,
        "fever_duration_days": fever_days,
        "diagnoses": _terms_present(blob, DIAGNOSIS_TERMS),
        "symptoms": _terms_present(blob, SYMPTOM_TERMS),
        "treatments": _terms_present(blob, TREATMENT_TERMS),
    }
    out.update(labs)
    return out


def _decode_barcodes(image: Image.Image) -> Dict[str, Any]:
    try:
        import zxingcpp  # type: ignore
    except Exception:
        return {"has_qr_code": 0, "has_barcode": 0, "codes": []}
    try:
        codes = zxingcpp.read_barcodes(np.array(image.convert("RGB")))
    except Exception:
        return {"has_qr_code": 0, "has_barcode": 0, "codes": []}
    out = []
    has_qr, has_bar = 0, 0
    for code in codes:
        fmt = str(getattr(code, "format", "") or "").upper()
        text = str(getattr(code, "text", "") or "")
        if "QR" in fmt:
            has_qr = 1
        elif fmt:
            has_bar = 1
        out.append({"format": fmt, "text": text[:200]})
    return {"has_qr_code": has_qr, "has_barcode": has_bar, "codes": out}


def detect_visual_elements(
    page_image: Image.Image,
    extracted_text: str = "",
    vlm_payload: Optional[Dict[str, Any]] = None,
    file_name: Optional[str] = None,
) -> Dict[str, Any]:
    file_name = file_name or ""
    lo = f"{file_name} {extracted_text}".lower()
    vlm = (vlm_payload or {}).get("visual_elements") or {}
    codes = _decode_barcodes(page_image)
    has_xray = int(bool(vlm.get("has_xray")) or _is_xray_like(file_name, extracted_text))
    has_invoice = any(x in lo for x in ["barcode", "invoice", "sticker", "implant", "udi", "batch no", "lot no"])
    has_photo = any(x in lo for x in ["photo", "pic", "intra", "specimen", "post_op", "post-op", "clinical photograph"])
    return {
        "has_stamp": int(bool(vlm.get("has_stamp")) or "stamp" in lo),
        "has_signature": int(bool(vlm.get("has_signature")) or "signature" in lo or "signed" in lo),
        "has_photo_evidence": int(bool(vlm.get("has_photo_evidence")) or has_photo),
        "has_implant_sticker": int(bool(vlm.get("has_implant_sticker")) or has_invoice or bool(codes["has_barcode"])),
        "has_table": int(bool(vlm.get("has_table")) or len(re.findall(r"\s{2,}|\|", extracted_text or "")) > 10),
        "has_xray": has_xray,
        "has_qr_code": int(codes["has_qr_code"]),
        "has_barcode": int(codes["has_barcode"]),
        "barcode_values": codes.get("codes", []),
    }


def _normalized_name(file_name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", file_name.lower())


def filename_doc_type_hint(file_name: str, package_code: Optional[str] = None) -> Optional[str]:
    name = _normalized_name(file_name)
    if any(x in name for x in ["feedback", "aadhar", "adhar", "card", "id_proof", "declaration"]):
        return "extra_document"
    if package_code == "SB039A":
        if any(x in name for x in ["post_xray", "post_op_xray", "postoperative_xray"]):
            return "post_op_xray"
        if any(x in name for x in ["x_ray", "xray", "_x_", "ct_knee"]):
            return "xray_ct_knee"
        if any(x in name for x in ["barcode", "invoice", "implant"]):
            return "implant_invoice"
        if any(x in name for x in ["post_op_photo", "pic", "photo"]):
            return "post_op_photo"
        if any(x in name for x in ["ot_note", "operative", "operation", "_ot_"]):
            return "operative_notes"
        if any(x in name for x in ["initial_assessment", "clinical", "_cl_"]):
            return "clinical_notes"
        if any(x in name for x in ["discharge", "_dis_", "_dc_", "sds"]):
            return "discharge_summary"
    if package_code == "SG039C":
        if any(x in name for x in ["usg", "ultra_sound", "ultrasound", "sonography"]):
            return "usg_report"
        if "lft" in name or "rftlft" in name:
            return "lft_report"
        if any(x in name for x in ["anaesth", "anesth", "pac"]):
            return "pre_anesthesia"
        if any(x in name for x in ["histopath", "hpe", "biopsy", "pathology"]):
            return "histopathology"
        if any(x in name for x in ["operative", "ot_note", "otnotes", "_ot_"]):
            return "operative_notes"
        if any(x in name for x in ["intra", "specimen", "photo", "films", "mkp"]):
            return "photo_evidence"
        if any(x in name for x in ["clinical", "prea", "case_"]):
            return "clinical_notes"
        if any(x in name for x in ["discharge", "_dis_", "_dc_"]):
            return "discharge_summary"
    if package_code == "MG064A":
        if any(x in name for x in ["post_hb", "post_cbc", "repeat_cbc", "repeat_hb"]):
            return "post_hb_report"
        if any(x in name for x in ["cbc", "hb_report", "hbr", "lab_report", "all_reports", "print_report", "_report"]):
            return "cbc_hb_report"
        if any(x in name for x in ["treatment", "medication", "chart", "transfusion", "intake_output", "vitals"]):
            return "treatment_details"
        if any(x in name for x in ["ipd", "icp", "bht", "case", "indoor"]):
            return "indoor_case"
        if any(x in name for x in ["clinical", "notes", "justification"]):
            return "clinical_notes"
        if any(x in name for x in ["discharge", "_dis_", "disc"]):
            return "discharge_summary"
    if package_code == "MG006A":
        if any(x in name for x in ["repeat", "post_treatment", "post"]):
            if any(x in name for x in ["cbc", "esr", "lft", "widal", "culture", "invest"]):
                return "investigation_post"
        if any(x in name for x in ["widal", "typhi", "culture", "cbc", "esr", "lft", "urine", "investigation", "inves", "fever_profile"]):
            return "investigation_pre"
        if any(x in name for x in ["tpr", "vital", "doctor_notes", "progress", "nurse", "icu", "treatment", "medication"]):
            return "vitals_treatment"
        if any(x in name for x in ["case", "admission", "clinical", "opd", "_cn_"]):
            return "clinical_notes"
        if any(x in name for x in ["discharge", "_dis_", "_dc_", "sum"]):
            return "discharge_summary"
    return None


KEYWORD_DOC_HINTS = {
    "clinical_notes": ["chief complaint", "history", "examination", "clinical notes", "provisional diagnosis", "h/o"],
    "cbc_hb_report": ["complete blood count", "cbc", "haemoglobin", "hemoglobin", "hgb", "rbc", "wbc", "platelet"],
    "post_hb_report": ["post transfusion", "post-treatment", "repeat hb", "after transfusion"],
    "indoor_case": ["indoor case", "ipd", "bed head ticket", "bht", "ward", "case sheet"],
    "treatment_details": ["treatment chart", "medication chart", "blood transfusion", "prbc", "ferrous", "drug chart"],
    "discharge_summary": ["discharge summary", "discharge card", "date of discharge", "condition at discharge"],
    "usg_report": ["ultrasonography", "ultrasound", "usg", "sonography", "gall bladder"],
    "lft_report": ["liver function", "bilirubin", "sgot", "sgpt", "alkaline phosphatase", "lft"],
    "operative_notes": ["operative note", "operation note", "ot note", "procedure performed", "intra operative"],
    "pre_anesthesia": ["pre anaesthesia", "pre anesthesia", "pac", "fitness for surgery"],
    "histopathology": ["histopathology", "histopathological", "microscopic", "gross examination", "biopsy"],
    "xray_ct_knee": ["x-ray knee", "xray knee", "knee ap", "knee lateral", "ct knee"],
    "post_op_xray": ["post operative x-ray", "post-op x-ray", "implant in situ", "prosthesis"],
    "implant_invoice": ["implant invoice", "barcode", "sticker", "udi", "lot no", "batch no"],
    "post_op_photo": ["post operative photo", "post-op photo", "clinical photograph"],
    "photo_evidence": ["intraoperative photograph", "specimen", "gall bladder specimen", "clinical photograph"],
    "investigation_pre": ["widal", "typhidot", "blood culture", "cbc", "esr", "peripheral smear", "lft"],
    "investigation_post": ["repeat widal", "repeat cbc", "post treatment", "follow up investigation"],
    "vitals_treatment": ["temperature chart", "tpr", "vitals", "pulse", "respiration", "doctor notes", "nursing notes"],
}


def classify_document_type(
    package_code_or_text: Any = None,
    extracted_text: Optional[str] = None,
    visual_tags: Optional[Dict[str, Any]] = None,
    vlm_payload: Optional[Dict[str, Any]] = None,
    file_name: Optional[str] = None,
) -> Dict[str, Any]:
    package_code = package_code_or_text if package_code_or_text in _nb("PACKAGE_CODES", []) else None
    if vlm_payload is None and isinstance(package_code_or_text, dict):
        vlm_payload = package_code_or_text
    text = extracted_text or (vlm_payload or {}).get("ocr_snippet", "") or ""
    visual_tags = visual_tags or (vlm_payload or {}).get("visual_elements") or {}
    document_types = set(_nb("DOCUMENT_TYPES", _nb("DOC_TYPE_VOCAB", [])))

    hint = filename_doc_type_hint(file_name or "", package_code)
    if hint and hint in document_types and hint != "extra_document":
        return {"doc_type": hint, "confidence": 0.93}
    if hint == "extra_document":
        return {"doc_type": "extra_document", "confidence": 0.90}

    if vlm_payload:
        dt = str(vlm_payload.get("doc_type", "")).lower()
        conf = float(vlm_payload.get("doc_type_confidence") or 0.0)
        if dt in document_types and conf >= 0.82:
            return {"doc_type": dt, "confidence": conf}

    lo = text.lower()
    scores: Dict[str, int] = defaultdict(int)
    for dt, terms in KEYWORD_DOC_HINTS.items():
        for term in terms:
            if term in lo:
                scores[dt] += 1
    if _rf_fuzz and text and (not scores or max(scores.values()) <= 1):
        blob = lo[:6000]
        for dt, terms in KEYWORD_DOC_HINTS.items():
            for term in terms:
                if len(term) > 4 and _rf_fuzz.partial_ratio(term, blob) >= 86:
                    scores[dt] = scores.get(dt, 0) + 1
    if visual_tags.get("has_xray"):
        scores["post_op_xray" if "post" in (file_name or "").lower() else "xray_ct_knee"] += 3
    if visual_tags.get("has_barcode") or visual_tags.get("has_implant_sticker"):
        scores["implant_invoice"] += 4
    if visual_tags.get("has_photo_evidence"):
        scores["photo_evidence"] += 2

    if scores:
        best, score = max(scores.items(), key=lambda item: item[1])
        if best in document_types and score > 0:
            return {"doc_type": best, "confidence": min(0.35 + 0.12 * score, 0.88)}
    return {"doc_type": "extra_document", "confidence": 0.20}


def _merge_entities(det: Dict[str, Any], llm: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    llm_ent = (llm or {}).get("entities") or {}
    merged = dict(det)
    for key in ["dates_found", "hb_values", "diagnoses", "symptoms", "treatments", "peripheral_smear_findings"]:
        vals: List[Any] = []
        for source in (det.get(key, []), llm_ent.get(key, [])):
            for item in source or []:
                if item not in vals:
                    vals.append(item)
        merged[key] = vals
    scalar_keys = [
        "patient_age", "doa", "dod", "temperature_celsius", "fever_duration_days",
        "wbc", "rbc", "platelet", "hematocrit", "esr_value",
        "bilirubin_total", "sgot_ast", "sgpt_alt", "alkaline_phosphatase",
        "laterality", "follow_up_date",
    ]
    for key in scalar_keys:
        merged[key] = det.get(key) if det.get(key) not in (None, "", []) else llm_ent.get(key)
    return merged


def _should_call_gemma(package_code: str, doc_type: str, doc_conf: float, text: str, visual: Dict[str, Any], file_name: str) -> bool:
    if not _nb("USE_GEMMA_FALLBACK", True):
        return False
    if _nb("USE_GEMMA_FOR_ALL_PAGES", False):
        return True
    if len((text or "").strip()) < 80:
        return True
    if doc_conf < float(_nb("MIN_GEMMA_DOC_CONF", 0.72)):
        return True
    if visual.get("has_xray") or visual.get("has_photo_evidence") or visual.get("has_barcode"):
        return True
    if package_code in {"SB039A", "SG039C"} and doc_type in {"post_op_xray", "xray_ct_knee", "photo_evidence", "implant_invoice"}:
        return True
    return False


def _gemma_payload(page_image: Image.Image, case_id: str, file_name: str, page_number: int, package_code: str) -> Dict[str, Any]:
    if not callable(_ORIGINAL_ANALYZE_PAGE_WITH_GEMMA):
        return {}
    try:
        return _ORIGINAL_ANALYZE_PAGE_WITH_GEMMA(page_image, case_id, file_name, page_number, package_code) or {}
    except Exception as exc:
        print(f"[gemma-selective] {case_id}/{file_name} p{page_number} failed: {exc}")
        return {}


def _clinical_payload(page_image: Image.Image, case_id: str, file_name: str, page_number: int, package_code: str, used_gemma: bool) -> Dict[str, int]:
    if not _nb("USE_CLINICAL_GEMMA", False) or not used_gemma:
        return {}
    if not callable(_ORIGINAL_ANALYZE_CLINICAL_WITH_GEMMA):
        return {}
    try:
        return _ORIGINAL_ANALYZE_CLINICAL_WITH_GEMMA(page_image, case_id, file_name, page_number, package_code) or {}
    except Exception:
        return {}


def _payload_for_page(text: str, doc_type: str, doc_conf: float, det_entities: Dict[str, Any], visual: Dict[str, Any], llm_payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    llm_payload = llm_payload or {}
    llm_text = llm_payload.get("ocr_snippet") or ""
    merged_text = text if len(text) >= len(llm_text) else f"{text}\n{llm_text}".strip()
    merged_entities = _merge_entities(det_entities, llm_payload)
    if llm_payload.get("doc_type_confidence", 0) and float(llm_payload.get("doc_type_confidence") or 0) > doc_conf:
        doc_type = llm_payload.get("doc_type") or doc_type
        doc_conf = float(llm_payload.get("doc_type_confidence") or doc_conf)
    return {
        "doc_type": doc_type,
        "doc_type_confidence": doc_conf,
        "ocr_snippet": merged_text,
        "language": llm_payload.get("language") or "en",
        "is_blurry": int(bool(llm_payload.get("is_blurry"))),
        "visual_elements": {**visual, **(llm_payload.get("visual_elements") or {})},
        "entities": merged_entities,
    }


def _dedupe_staged_by_dhash(staged: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not bool(_nb("DEDUPE_PAGES_BY_DHASH", True)) or not staged:
        return staged
    out: List[Dict[str, Any]] = []
    last_snip: Dict[str, str] = {}
    for item in staged:
        dh = str(item.get("_dhash", ""))
        snip = ((item.get("ocr_text") or "")[:500]).lower()
        if dh:
            prev = last_snip.get(dh)
            if prev is not None:
                if _rf_fuzz is not None:
                    try:
                        if _rf_fuzz.ratio(prev, snip) > 94:
                            continue
                    except Exception:
                        if prev[:200] == snip[:200]:
                            continue
                elif prev[:200] == snip[:200]:
                    continue
            last_snip[dh] = snip
        out.append(item)
    return out


def dedupe_page_results_by_dhash(page_results: List[Any]) -> List[Any]:
    """Dedupe a list of PageResult-like objects using ``evidence.page_dhash`` + text similarity."""
    if not bool(_nb("DEDUPE_PAGES_BY_DHASH", True)) or not page_results:
        return page_results
    out: List[Any] = []
    last_snip: Dict[str, str] = {}
    for pr in page_results:
        ev = getattr(pr, "evidence", None) or {}
        dh = str(ev.get("page_dhash") or "")
        snip = ((getattr(pr, "extracted_text", "") or "")[:500]).lower()
        if dh:
            prev = last_snip.get(dh)
            if prev is not None:
                if _rf_fuzz is not None:
                    try:
                        if _rf_fuzz.ratio(prev, snip) > 94:
                            continue
                    except Exception:
                        if prev[:200] == snip[:200]:
                            continue
                elif prev[:200] == snip[:200]:
                    continue
            last_snip[dh] = snip
        out.append(pr)
    return out


def build_claim_evidence_bundle(
    case_id: str,
    package_code: str,
    page_results: Sequence[Any],
    strict_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compact claim-level JSON for step-2 LLM (no raw images)."""
    doc_counts: Dict[str, int] = defaultdict(int)
    for pr in page_results or []:
        dt = str(getattr(pr, "doc_type", "") or "")
        doc_counts[dt] += 1
    agg = _aggregate_rows(list(strict_rows or []))
    flag_keys = [
        "severe_anemia", "fever", "usg_calculi", "pain_present", "previous_surgery",
        "post_op_implant_present", "age_valid", "clinical_condition",
    ]
    flags = {k: agg.get(k) for k in flag_keys if agg.get(k) not in (None, 0, "", False)}
    snippets: List[Dict[str, Any]] = []
    for pr in (page_results or [])[:48]:
        head = ((getattr(pr, "extracted_text", "") or "").replace("\n", " "))[:260]
        snippets.append({
            "file": getattr(pr, "file_name", ""),
            "page": getattr(pr, "page_number", 0),
            "doc_type": getattr(pr, "doc_type", ""),
            "text_head": head,
        })
    return {
        "case_id": case_id,
        "package_code": package_code,
        "doc_type_counts": dict(doc_counts),
        "aggregated_rule_flags": flags,
        "nullable_dates": {k: agg.get(k) for k in _nb("NULLABLE_DATE_KEYS", set()) if agg.get(k)},
        "page_snippets": snippets,
        "page_total": len(page_results or []),
    }


def _claim_level_reconciliation(case_id: str, package_code: str, bundle: Dict[str, Any]) -> Dict[str, Any]:
    if not bool(_nb("USE_CLAIM_LEVEL_GEMMA", True)):
        return {}
    if not callable(_ORIGINAL_ANALYZE_CLAIM_RECONCILE):
        return {}
    try:
        out = _ORIGINAL_ANALYZE_CLAIM_RECONCILE(case_id, package_code, bundle)
        return out if isinstance(out, dict) else {}
    except Exception as exc:
        print(f"[claim-reconcile] {case_id}: {exc}")
        return {}


def process_case(case_id: str, files: List[Path], package_code: str) -> Dict[str, Any]:
    PageResult = _nb("PageResult")
    if PageResult is None:
        raise RuntimeError("PageResult dataclass not found in notebook globals")

    staged: List[Dict[str, Any]] = []

    for fp in files:
        for page in extract_pages(fp):
            page_image: Image.Image = page["image"]
            page_number = int(page["page_number"])
            file_name = page["file_name"]
            embedded_text = page.get("embedded_text", "")

            ocr_text, ocr_lines = run_ocr(
                page_image,
                case_id=case_id,
                file_name=file_name,
                page_number=page_number,
                embedded_text=embedded_text,
            )
            det_entities = extract_deterministic_entities(ocr_text, file_name=file_name, ocr_lines=ocr_lines)
            prelim_visual = detect_visual_elements(page_image, ocr_text, file_name=file_name)
            prelim_cls = classify_document_type(package_code, ocr_text, prelim_visual, file_name=file_name)
            doc_type = prelim_cls["doc_type"]
            doc_conf = float(prelim_cls["confidence"])

            used_gemma = False
            llm_payload: Dict[str, Any] = {}
            if _should_call_gemma(package_code, doc_type, doc_conf, ocr_text, prelim_visual, file_name):
                llm_payload = _gemma_payload(page_image, case_id, file_name, page_number, package_code)
                used_gemma = bool(llm_payload)

            visual = detect_visual_elements(page_image, ocr_text, vlm_payload=llm_payload, file_name=file_name)
            cls = classify_document_type(package_code, ocr_text, visual, vlm_payload=llm_payload, file_name=file_name)
            payload = _payload_for_page(
                text=ocr_text,
                doc_type=cls["doc_type"],
                doc_conf=float(cls["confidence"]),
                det_entities=det_entities,
                visual=visual,
                llm_payload=llm_payload,
            )
            quality = _nb("estimate_page_quality")(page_image, ocr_text) if callable(_nb("estimate_page_quality")) else {}
            clinical_payload = _clinical_payload(page_image, case_id, file_name, page_number, package_code, used_gemma)

            staged.append({
                "_dhash": difference_hash16(page_image),
                "page_number": page_number,
                "file_name": file_name,
                "ocr_text": ocr_text,
                "ocr_lines": ocr_lines,
                "embedded_text": embedded_text,
                "payload": payload,
                "visual": visual,
                "quality": quality,
                "clinical_payload": clinical_payload,
                "used_gemma": used_gemma,
                "llm_payload": llm_payload,
            })

    staged = _dedupe_staged_by_dhash(staged)

    page_results: List[Any] = []
    payloads_for_dates: List[Dict[str, Any]] = []
    for st in staged:
        payload = st["payload"]
        pr = PageResult(
            case_id=case_id,
            file_name=st["file_name"],
            page_number=int(st["page_number"]),
            extracted_text=payload.get("ocr_snippet", "") or "",
            ocr_lines=st["ocr_lines"],
            doc_type=payload.get("doc_type", "extra_document"),
            doc_type_confidence=float(payload.get("doc_type_confidence") or 0.0),
            visual_tags=st["visual"],
            entities={"entities": payload.get("entities") or {}, "ocr_snippet": payload.get("ocr_snippet", "")},
            quality=st.get("quality") or {},
            output_row={},
            evidence={
                "ocr_engine": "PaddleOCR",
                "ocr_line_count": len(st["ocr_lines"]),
                "embedded_text_chars": len(st.get("embedded_text") or ""),
                "gemma_used": bool(st.get("used_gemma")),
                "vlm_language": payload.get("language"),
                "vlm_doc_conf": (st.get("llm_payload") or {}).get("doc_type_confidence"),
                "clinical_payload": st.get("clinical_payload") or {},
                "page_dhash": st.get("_dhash"),
            },
        )
        page_results.append(pr)
        payloads_for_dates.append({"entities": payload.get("entities") or {}})

    strict_rows: List[Dict[str, Any]] = []
    for pr in page_results:
        row = _nb("populate_row_for_package")(package_code, pr)
        pr.output_row = row
        strict_rows.append(row)

    if callable(_nb("_propagate_case_clinical")):
        strict_rows = _nb("_propagate_case_clinical")(package_code, strict_rows)
    if package_code == "SG039C":
        if callable(_nb("_sg039c_force_extra_by_filename")):
            strict_rows = _nb("_sg039c_force_extra_by_filename")(strict_rows)
        if callable(_nb("_sg039c_multi_page_continuity")):
            strict_rows = _nb("_sg039c_multi_page_continuity")(strict_rows)
    if callable(_nb("assign_document_ranks")):
        strict_rows = _nb("assign_document_ranks")(package_code, strict_rows)
    if callable(_nb("_propagate_case_dates")):
        strict_rows = _nb("_propagate_case_dates")(package_code, strict_rows, payloads_for_dates)
    if callable(_nb("normalize_dates_in_rows")):
        strict_rows = _nb("normalize_dates_in_rows")(package_code, strict_rows)

    claim_bundle = build_claim_evidence_bundle(case_id, package_code, page_results, strict_rows)
    claim_recon = _claim_level_reconciliation(case_id, package_code, claim_bundle)

    timeline = _nb("build_episode_timeline")(package_code, page_results) if callable(_nb("build_episode_timeline")) else []
    decision = run_rules_engine(
        case_id, package_code, strict_rows, timeline,
        page_results=page_results,
        claim_reconciliation=claim_recon,
    )
    summary_df = _nb("build_human_readable_summary")(package_code, page_results, decision) if callable(_nb("build_human_readable_summary")) else None
    timeline_df = _nb("build_timeline_df")(timeline) if callable(_nb("build_timeline_df")) else None

    return {
        "case_id": case_id,
        "package_code": package_code,
        "page_results": page_results,
        "strict_rows": strict_rows,
        "timeline": timeline,
        "decision": decision,
        "summary_df": summary_df,
        "timeline_df": timeline_df,
        "claim_evidence_bundle": claim_bundle,
        "claim_reconciliation": claim_recon,
    }


def _case_text(page_results: Optional[List[Any]]) -> str:
    return "\n".join((getattr(pr, "extracted_text", "") or "") for pr in (page_results or []))


def _texts_by_doc(page_results: Optional[List[Any]], doc_type: str) -> str:
    return "\n".join((getattr(pr, "extracted_text", "") or "") for pr in (page_results or []) if getattr(pr, "doc_type", "") == doc_type)


def _aggregate_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    agg: Dict[str, Any] = {}
    nullable = _nb("NULLABLE_DATE_KEYS", set())
    for row in rows:
        for key, val in row.items():
            if key in nullable:
                if val and not agg.get(key):
                    agg[key] = val
            elif isinstance(val, (int, float)):
                agg[key] = max(int(agg.get(key, 0) or 0), int(val))
            else:
                agg.setdefault(key, val)
    return agg


def _contains(text: str, terms: Sequence[str]) -> bool:
    lo = (text or "").lower()
    return any(term in lo for term in terms)


def _has_lft_elevation(text: str) -> bool:
    lo = (text or "").lower()
    markers = ["bilirubin", "sgot", "sgpt", "ast", "alt", "alkaline phosphatase", "alp", "transaminase"]
    if not any(marker in lo for marker in markers):
        return False
    if any(word in lo for word in ["elevated", "high", "raised", "increased"]):
        return True
    numeric_rules = [
        (r"\bbilirubin\s*[:=\-]?\s*(\d+(?:\.\d+)?)", 1.3),
        (r"\b(?:sgot|ast)\s*[:=\-]?\s*(\d+(?:\.\d+)?)", 45.0),
        (r"\b(?:sgpt|alt)\s*[:=\-]?\s*(\d+(?:\.\d+)?)", 45.0),
        (r"\b(?:alkaline phosphatase|alp)\s*[:=\-]?\s*(\d+(?:\.\d+)?)", 130.0),
    ]
    for pat, threshold in numeric_rules:
        for match in re.finditer(pat, lo):
            try:
                if float(match.group(1)) > threshold:
                    return True
            except Exception:
                pass
    return False


def _alos_reason(package_code: str, doa: Optional[str], dod: Optional[str]) -> Optional[str]:
    if not doa or not dod:
        return None
    limits = _nb("ALOS_LIMITS", {"MG064A": 3, "SG039C": 3, "MG006A": 5, "SB039A": 7})
    limit = limits.get(package_code)
    if not limit:
        return None
    try:
        d1 = datetime.strptime(_normalize_date(doa) or doa, "%d-%m-%Y")
        d2 = datetime.strptime(_normalize_date(dod) or dod, "%d-%m-%Y")
        los = (d2 - d1).days
        if los > limit:
            return f"Length of stay ({los}d) exceeds ALOS limit ({limit}d)"
    except Exception:
        return None
    return None


def run_rules_engine(
    case_id: str,
    package_code: str,
    rows: List[Dict[str, Any]],
    timeline: List[Any],
    page_results: Optional[List[Any]] = None,
    claim_reconciliation: Optional[Dict[str, Any]] = None,
) -> Any:
    ClaimDecision = _nb("ClaimDecision")
    if ClaimDecision is None:
        raise RuntimeError("ClaimDecision dataclass not found in notebook globals")

    case = _aggregate_rows(rows)
    all_text = _case_text(page_results)
    reasons: List[str] = []
    rule_flags: List[str] = []
    timeline_flags: List[str] = []
    claim_reconciliation = claim_reconciliation or {}
    for f in claim_reconciliation.get("suggested_timeline_flags") or []:
        if isinstance(f, str) and f.strip():
            timeline_flags.append(f.strip())
    for note in claim_reconciliation.get("manual_review_notes") or []:
        if isinstance(note, str) and note.strip():
            reasons.append(f"[claim-review] {note.strip()}")
    mandatory = _nb("MANDATORY_DOCS", {}).get(package_code, [])
    missing_docs = [doc for doc in mandatory if not case.get(doc)]
    for doc in missing_docs:
        reasons.append(f"Missing mandatory document: {doc}")

    if package_code == "MG064A":
        pre_text = _texts_by_doc(page_results, "cbc_hb_report") or all_text
        treatment_text = _texts_by_doc(page_results, "treatment_details") or all_text
        pre_hbs = _hb_values(pre_text)
        if not any(v < 7.0 for v in pre_hbs) and not case.get("severe_anemia"):
            rule_flags.append("TMS_RULE_1_SEVERITY")
            reasons.append("Pre-treatment hemoglobin <7 g/dL not found.")
        transfusion = _contains(treatment_text, ["blood transfusion", "blood transfused", "prbc", "packed red blood", "whole blood"])
        iron = _contains(treatment_text, ["ferrous sulphate", "ferrous sulfate", "iron injection", "iv iron", "parenteral iron"])
        if not (transfusion and iron):
            rule_flags.append("TMS_RULE_2_TREATMENT")
            missing = []
            if not transfusion:
                missing.append("blood transfusion")
            if not iron:
                missing.append("iron/ferrous injection")
            reasons.append("Mandatory anemia treatment not documented: " + ", ".join(missing))
        post_text = _texts_by_doc(page_results, "post_hb_report")
        post_hbs = _hb_values(post_text)
        if case.get("post_hb_report") and not post_hbs:
            timeline_flags.append("Post-treatment Hb numeric value not confidently extracted from post_hb_report pages")

    elif package_code == "MG006A":
        temp_c, fever_days = _temperature_and_duration(all_text)
        fever_ok = bool(temp_c is not None and temp_c >= 38.3 and fever_days is not None and fever_days > 2)
        if not fever_ok:
            rule_flags.append("TMS_RULE_1_FEVER")
            reasons.append("Fever threshold/duration not proven: need >=38.3 C / 101 F and >2 days.")
        if not case.get("investigation_pre"):
            rule_flags.append("TMS_RULE_2_PRE_LAB")
        if not case.get("investigation_post"):
            rule_flags.append("TMS_RULE_2_POST_LAB")
        if _contains(all_text, ["organ dysfunction", "sepsis", "septic shock", "multi-organ failure", "organ failure"]):
            timeline_flags.append("Exclusion criteria present: organ dysfunction/sepsis requires manual review")

    elif package_code == "SG039C":
        usg_text = _texts_by_doc(page_results, "usg_report") or all_text
        lft_text = _texts_by_doc(page_results, "lft_report") or all_text
        if not case.get("usg_calculi") and not _contains(usg_text, ["calculi", "calculus", "gall stone", "gallstone", "cholelithiasis"]):
            rule_flags.append("TMS_RULE_1_IMAGING")
            reasons.append("USG evidence of gall bladder calculi not found.")
        if not case.get("pain_present") and not _contains(all_text, ["right hypochondrium", "epigastrium", "ruq", "biliary colic", "abdominal pain"]):
            rule_flags.append("TMS_RULE_2_CLINICAL")
            reasons.append("Mandatory pain symptoms not found.")
        if case.get("previous_surgery") or _contains(all_text, ["prior cholecystectomy", "previous cholecystectomy", "history of cholecystectomy"]):
            rule_flags.append("TMS_RULE_3_FRAUD")
            reasons.append("Prior cholecystectomy history found.")
        if not _has_lft_elevation(lft_text):
            rule_flags.append("TMS_RULE_4_LFT")
            reasons.append("Elevated LFT marker evidence not found.")

    elif package_code == "SB039A":
        if not case.get("post_op_implant_present"):
            rule_flags.append("TMS_RULE_1_POST_OP_VISUAL")
            reasons.append("Post-operative X-ray implant evidence not found.")
        if not case.get("age_valid"):
            rule_flags.append("TMS_RULE_3_AGE_CHECK")
            reasons.append("Primary TKR age criterion (>55 or waived by trauma/systemic disease) not proven.")
        if _contains(all_text, ["active infection", "infected joint", "septic arthritis", "wound infection", "evidence of infection"]):
            rule_flags.append("TMS_RULE_4_INFECTION")
            reasons.append("Infection exclusion evidence found.")
        post_xray_text = _texts_by_doc(page_results, "post_op_xray")
        if case.get("post_op_xray") and not _contains(post_xray_text, ["left", "right", " lt", " rt", " l ", " r "]):
            timeline_flags.append("Post-op X-ray laterality label not confidently extracted")
        if case.get("post_op_xray") and not _find_dates(post_xray_text):
            timeline_flags.append("Post-op X-ray date not confidently extracted")

    alos = _alos_reason(package_code, case.get("doa") or case.get("pre_date"), case.get("dod") or case.get("post_date"))
    if alos:
        timeline_flags.append(alos)

    critical_failures = {
        "TMS_RULE_1_SEVERITY",
        "TMS_RULE_2_TREATMENT",
        "TMS_RULE_1_FEVER",
        "TMS_RULE_1_IMAGING",
        "TMS_RULE_2_CLINICAL",
        "TMS_RULE_3_FRAUD",
        "TMS_RULE_4_LFT",
        "TMS_RULE_1_POST_OP_VISUAL",
        "TMS_RULE_3_AGE_CHECK",
        "TMS_RULE_4_INFECTION",
    }
    if any(flag in critical_failures for flag in rule_flags):
        decision, confidence = _nb("DECISION_FAIL", "FAIL"), 0.90
    elif missing_docs or rule_flags or timeline_flags:
        decision, confidence = _nb("DECISION_CONDITIONAL", "CONDITIONAL"), 0.75
    else:
        decision, confidence = _nb("DECISION_PASS", "PASS"), 0.95
        reasons = ["All mandatory documents present", "All STG rules passed", "Timeline validated"]

    return ClaimDecision(
        case_id=case_id,
        package_code=package_code,
        decision=decision,
        confidence=confidence,
        reasons=reasons,
        missing_documents=missing_docs,
        rule_flags=rule_flags,
        timeline_flags=timeline_flags,
        claim_reconciliation=claim_reconciliation if claim_reconciliation else None,
    )
