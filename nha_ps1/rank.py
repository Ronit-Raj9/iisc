"""document_rank assignment with multi-page-PDF consistency.

HI.txt §A.6 note: 'If a document spans multiple pages... the same document_rank
must be assigned to all pages of that document.'
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List

from .schemas import LINK_FIELD, RANK_MAP


def _per_page_rank(package_code: str, doc_type: str, extra_doc: int) -> int:
    if extra_doc:
        return 99
    return RANK_MAP.get(package_code, {}).get(doc_type, 99)


def assign_ranks(package_code: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Set row['document_rank'] consistently per source file.

    Strategy:
      - compute a candidate rank for every page
      - group pages by their `link` field (= source file path/name)
      - for each group, take the most common non-99 rank as the canonical rank
        (ties broken by the smallest rank — earlier in timeline)
      - if a group is entirely 'extra_document', stays at 99
    """
    link_key = LINK_FIELD[package_code]

    candidates: List[int] = []
    for row in rows:
        candidates.append(
            _per_page_rank(
                package_code,
                _doc_type_from_row(package_code, row),
                int(row.get("extra_document", 0) or 0),
            )
        )

    # Group page-indices by source file.
    groups: Dict[str, List[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        groups[str(row.get(link_key, ""))].append(idx)

    canonical: Dict[str, int] = {}
    for link, idxs in groups.items():
        ranks_in_group = [candidates[i] for i in idxs]
        non_extra = [r for r in ranks_in_group if r != 99]
        if not non_extra:
            canonical[link] = 99
            continue
        counter = Counter(non_extra)
        # Most-common; tie-break to smallest rank for deterministic ordering.
        most_common = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        canonical[link] = most_common[0][0]

    for idx, row in enumerate(rows):
        link = str(row.get(link_key, ""))
        # An extra page in an otherwise-classified PDF: keep its 99
        # only if THIS page is genuinely extra. Per HI.txt §A.6 note:
        # "all pages belonging to that investigation PDF must have document_rank = 3".
        # So if the PDF as a whole is classified, every page inherits the rank.
        if int(row.get("extra_document", 0) or 0) and candidates[idx] == 99 and canonical[link] == 99:
            row["document_rank"] = 99
        else:
            row["document_rank"] = canonical[link]
    return rows


def _doc_type_from_row(package_code: str, row: Dict[str, Any]) -> str:
    """Recover the page's doc_type from its presence flags.

    The row only has the binary flags; we pick the highest-priority flag that
    is set to 1 (using RANK_MAP as a tie-breaker — lower rank wins).
    """
    candidates = [
        k for k, v in row.items()
        if k in RANK_MAP.get(package_code, {}) and v == 1
    ]
    if not candidates:
        return "extra_document" if row.get("extra_document", 0) else "extra_document"
    return min(candidates, key=lambda k: RANK_MAP[package_code][k])
