"""Local Hugging Face Gemma 3 multimodal client with OpenRouter-compatible ``.completion()`` API.

Use on Colab Pro / H100 (or any CUDA machine) to avoid OpenRouter latency and cost.
Requires: transformers>=4.50, torch, accelerate, pillow; HF token if the model is gated.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Union

MessageContent = Union[str, List[Dict[str, Any]]]


def _openai_messages_to_gemma3(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI-style chat (image_url + text parts) to Gemma3 processor format."""
    out: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role") or "user"
        c = m.get("content")
        if isinstance(c, str):
            out.append({"role": role, "content": [{"type": "text", "text": c}]})
            continue
        if not isinstance(c, list):
            out.append({"role": role, "content": [{"type": "text", "text": str(c)}]})
            continue
        parts: List[Dict[str, Any]] = []
        for part in c:
            if not isinstance(part, dict):
                continue
            typ = part.get("type")
            if typ == "text":
                parts.append({"type": "text", "text": str(part.get("text", ""))})
            elif typ == "image_url":
                url = (part.get("image_url") or {}).get("url") or ""
                if url:
                    parts.append({"type": "image", "url": str(url)})
            elif typ == "image":
                u = part.get("url")
                if u:
                    parts.append({"type": "image", "url": str(u)})
        if not parts:
            parts = [{"type": "text", "text": ""}]
        out.append({"role": role, "content": parts})
    return out


class LocalGemmaHFClient:
    """Loads ``Gemma3ForConditionalGeneration`` once; returns OpenAI-shaped ``completion`` dict."""

    def __init__(
        self,
        model_id: str = "google/gemma-3-12b-it",
        *,
        torch_dtype: str = "bfloat16",
        device_map: str = "auto",
        attn_implementation: str = "sdpa",
    ) -> None:
        import torch
        from transformers import AutoProcessor, Gemma3ForConditionalGeneration

        self.model_id = model_id
        if torch_dtype == "bfloat16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dt = torch.bfloat16
        elif torch_dtype == "bfloat16" and torch.cuda.is_available():
            dt = torch.float16
        elif torch_dtype == "float16" and torch.cuda.is_available():
            dt = torch.float16
        else:
            dt = torch.float32

        self._processor = AutoProcessor.from_pretrained(model_id, padding_side="left")
        self._model = Gemma3ForConditionalGeneration.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=dt,
            attn_implementation=attn_implementation,
        )
        self._torch = torch

    def completion(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del metadata
        max_new = int(max_tokens if max_tokens is not None else kwargs.get("max_tokens") or 1024)
        temp = float(temperature if temperature is not None else kwargs.get("temperature") or 0.15)

        gemma_messages = _openai_messages_to_gemma3(messages)
        inputs = self._processor.apply_chat_template(
            gemma_messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        )
        inputs = inputs.to(self._model.device)

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max(32, min(max_new, 8192)),
            "do_sample": temp > 0,
        }
        if temp > 0:
            gen_kwargs["temperature"] = min(max(temp, 0.01), 1.5)

        with self._torch.inference_mode():
            out = self._model.generate(**inputs, **gen_kwargs)

        prompt_len = int(inputs["input_ids"].shape[1])
        new_ids = out[0, prompt_len:]
        text = self._processor.decode(new_ids, skip_special_tokens=True)

        return {"choices": [{"message": {"role": "assistant", "content": text}}], "model": model or self.model_id}
