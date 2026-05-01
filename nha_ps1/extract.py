"""Entity-extraction helpers: dates, ages, Hb values, keyword presence."""

from __future__ import annotations

import re
from datetime import datetime
from typing import List, Optional

DATE_PATTERNS = [
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
    r"\b\d{1,2}[.]\d{1,2}[.]\d{2,4}\b",
    r"\b\d{1,2}-[A-Za-z]{3}-\d{2,4}\b",
    r"\b\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4}\b",
    r"\b[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{2,4}\b",
]

DATE_FORMATS = [
    "%d/%m/%y", "%d/%m/%Y",
    "%d-%m-%y", "%d-%m-%Y",
    "%d.%m.%y", "%d.%m.%Y",
    "%d-%b-%y", "%d-%b-%Y",
    "%d %b %Y", "%d %B %Y",
    "%d %b %y", "%d %B %y",
    "%b %d, %Y", "%B %d, %Y",
    "%b %d %Y", "%B %d %Y",
    "%m/%d/%y", "%m/%d/%Y",
]


def find_dates(text: str) -> List[str]:
    """Return unique date-shaped substrings from `text`, preserving order."""
    if not text:
        return []
    seen: set[str] = set()
    out: List[str] = []
    for pattern in DATE_PATTERNS:
        for m in re.findall(pattern, text):
            if m not in seen:
                seen.add(m)
                out.append(m)
    return out


def normalize_date(date_str: Optional[str]) -> Optional[str]:
    """Coerce a date string to DD-MM-YYYY (HI.txt §C.2 / D.2). Returns None on
    empty input; returns original string if no format matches."""
    if not date_str:
        return None
    s = str(date_str).strip()
    for fmt in DATE_FORMATS:
        try:
            dt = datetime.strptime(s, fmt)
            # Reject obviously bogus century rollovers from %y.
            if dt.year < 1950:
                dt = dt.replace(year=dt.year + 100)
            return dt.strftime("%d-%m-%Y")
        except ValueError:
            continue
    return s


def find_age(text: str) -> Optional[int]:
    """Best-effort patient age extraction."""
    if not text:
        return None
    patterns = [
        r"age[:\s]+(\d{1,3})",
        r"(\d{1,3})\s*(?:years?|yrs?|y\.?o\.?|year-old)",
        r"\b(\d{1,3})\s*y\b",
    ]
    lo = text.lower()
    for p in patterns:
        m = re.search(p, lo)
        if m:
            age = int(m.group(1))
            if 0 < age < 120:
                return age
    return None


def parse_hb_values(text: str) -> List[float]:
    """Pull out hemoglobin readings like 'Hb 6.5', 'Haemoglobin: 5.8 g/dL'."""
    if not text:
        return []
    out: List[float] = []
    for m in re.finditer(
        r"\b(?:hb|hgb|h\.?b|haemoglobin|hemoglobin)\s*[:=\-]?\s*(\d{1,2}(?:\.\d{1,2})?)\b",
        text,
        flags=re.IGNORECASE,
    ):
        try:
            val = float(m.group(1))
            if 0 < val < 25:  # plausible Hb range g/dL
                out.append(val)
        except ValueError:
            continue
    return out


def contains_any(text: str, keywords: List[str]) -> int:
    """Return 1 if any keyword (case-insensitive) appears in text, else 0."""
    if not text or not keywords:
        return 0
    lo = text.lower()
    return int(any(kw.lower() in lo for kw in keywords))


def first_or_none(values: List[str]) -> Optional[str]:
    return values[0] if values else None
