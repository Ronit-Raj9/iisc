"""Deterministic visual-element detection: QR/barcode via pyzbar.

Stamp/signature/photo/implant_sticker/table flags come from the VLM's JSON
(see vlm/qwen_runner.py); this module only handles the easy deterministic ones.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from PIL import Image

try:
    from pyzbar.pyzbar import decode as _zbar_decode  # type: ignore
    _HAS_ZBAR = True
except Exception as _e:
    # On Windows, pyzbar's libzbar-64.dll depends on libiconv.dll from the
    # MSVC runtime; if missing, we silently degrade to "no QR/barcode detected".
    # Install the Visual C++ Redistributable to enable, OR `conda install pyzbar`
    # which bundles the deps.
    _HAS_ZBAR = False


def detect_codes(image: Image.Image) -> Dict[str, int]:
    """Return {has_qr_code, has_barcode} as 0/1.

    pyzbar is a wrapper around libzbar0; on Windows the `pyzbar` wheel ships
    its own DLL so this works without extra install steps.
    """
    if not _HAS_ZBAR:
        return {"has_qr_code": 0, "has_barcode": 0}

    arr = np.array(image.convert("RGB"))
    try:
        decoded = _zbar_decode(arr)
    except Exception:
        return {"has_qr_code": 0, "has_barcode": 0}

    has_qr = 0
    has_bar = 0
    for d in decoded:
        t = (d.type or "").upper()
        if t == "QRCODE":
            has_qr = 1
        elif t:
            has_bar = 1
    return {"has_qr_code": has_qr, "has_barcode": has_bar}


def merge_visual_signals(vlm_visual: Dict[str, int], code_visual: Dict[str, int]) -> Dict[str, int]:
    """Combine VLM-reported visual flags with deterministic QR/barcode flags."""
    keys = (
        "has_stamp", "has_signature", "has_photo_evidence",
        "has_implant_sticker", "has_table", "has_xray",
        "has_qr_code", "has_barcode",
    )
    merged: Dict[str, int] = {}
    for k in keys:
        merged[k] = int(bool(vlm_visual.get(k, 0))) | int(bool(code_visual.get(k, 0)))
    return merged
