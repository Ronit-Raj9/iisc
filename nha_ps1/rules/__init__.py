"""STG-aware per-package rule modules.

Each module exposes:
  apply(row, context) -> mutated row

`context` is a dict with: text, text_en, doc_type, visual_tags, quality,
extracted (the VLM `extracted` block), entire_case_text (joined text from
all pages of the same case — used for severity flags / age that may live
on a different page than the document being classified).
"""

from . import mg064a, sg039c, mg006a, sb039a

PACKAGE_RULE_FNS = {
    "MG064A": mg064a.apply,
    "SG039C": sg039c.apply,
    "MG006A": mg006a.apply,
    "SB039A": sb039a.apply,
}

__all__ = ["PACKAGE_RULE_FNS"]
