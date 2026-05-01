"""OpenRouter chat-completions client with an NHAclient-compatible `.completion()` API."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

DEFAULT_BASE = "https://openrouter.ai/api/v1"


def parse_json_content(text: str) -> Any:
    """Parse JSON from an LLM message: strict json.loads, then json-repair, then brace slice.

    Handles markdown fences (```json ... ```) and leading/trailing prose.
    Returns a dict on success; empty dict if nothing parseable.
    """
    if text is None:
        return {}
    t = str(text).strip()
    if not t:
        return {}

    # Strip ```json ... ``` or ``` ... ```
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9]*\s*", "", t)
        t = re.sub(r"\s*```\s*$", "", t).strip()

    def _loads(s: str) -> Any:
        return json.loads(s)

    for candidate in (t,):
        try:
            out = _loads(candidate)
            return out if isinstance(out, (dict, list)) else {}
        except Exception:
            pass

    s = t.find("{")
    e = t.rfind("}")
    if s != -1 and e > s:
        chunk = t[s : e + 1]
        try:
            out = _loads(chunk)
            return out if isinstance(out, (dict, list)) else {}
        except Exception:
            try:
                from json_repair import repair_json  # type: ignore

                repaired = repair_json(chunk)
                out = json.loads(repaired)
                return out if isinstance(out, (dict, list)) else {}
            except Exception:
                pass

    try:
        from json_repair import repair_json  # type: ignore

        repaired = repair_json(t)
        out = json.loads(repaired)
        return out if isinstance(out, (dict, list)) else {}
    except Exception:
        return {}


class OpenRouterClient:
    """POSTs to OpenRouter; returns the same JSON shape as OpenAI (`choices` / `message` / `content`)."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: Optional[str] = None,
        site_url: str = "http://localhost",
        site_name: str = "iisc-ps1-local",
        timeout_s: Optional[float] = None,
    ) -> None:
        self.api_key = (api_key or "").strip()
        self.base_url = (base_url or os.environ.get("OPENROUTER_BASE_URL") or DEFAULT_BASE).rstrip("/")
        self.site_url = os.environ.get("OPENROUTER_SITE_URL", site_url)
        self.site_name = os.environ.get("OPENROUTER_SITE_NAME", site_name)
        raw_t = timeout_s if timeout_s is not None else os.environ.get("OPENROUTER_TIMEOUT", "600")
        self.timeout_s = float(raw_t)

    def completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del metadata  # NHA-only hint; OpenRouter ignores
        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is empty. Jupyter often ignores shell `export`. "
                "Use a repo-root `.env` (see `.env.example`, loaded by python-dotenv in the notebook), "
                "or assign OPENROUTER_API_KEY in the OpenRouter cell for this session only — never commit keys."
            )
        body: Dict[str, Any] = {"model": model, "messages": messages}
        if temperature is not None:
            body["temperature"] = temperature
        elif kwargs.get("temperature") is not None:
            body["temperature"] = kwargs["temperature"]
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        elif kwargs.get("max_tokens") is not None:
            body["max_tokens"] = kwargs["max_tokens"]

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": self.site_url,
                "X-Title": self.site_name,
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter HTTP {e.code}: {detail[:4000]}") from e
