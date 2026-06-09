"""Small OpenAI-compatible chat-completion client.

Most AMD AI Developer Cloud deployments expose either:

* a standard OpenAI-compatible vLLM/AIM path such as `/v1/chat/completions`;
* an AMD Workbench-style inference path such as `/v1/inference`.

This module supports both through `AI_API_STYLE=openai` or
`AI_API_STYLE=amd_inference`.
"""
from __future__ import annotations

import time
from typing import Any
from urllib.parse import urljoin

import httpx
from openai import OpenAI

from .llm_config import LLMSettings

_TIMEOUT = httpx.Timeout(180.0, connect=20.0)


def chat_completion(
    settings: LLMSettings,
    messages: list[dict[str, Any]],
    *,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """Call the configured chat endpoint and return `(text, usage_meta)`."""
    if settings.api_style == "amd_inference":
        return _amd_inference_completion(
            settings,
            messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=seed,
        )
    return _openai_completion(
        settings,
        messages,
        model=model,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
    )


def _openai_completion(
    settings: LLMSettings,
    messages: list[dict[str, Any]],
    *,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int | None,
) -> tuple[str, dict[str, Any]]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        kwargs["seed"] = seed

    client = OpenAI(
        api_key=settings.api_key or "EMPTY",
        base_url=settings.base_url,
        timeout=_TIMEOUT,
    )
    t0 = time.perf_counter()
    resp = client.chat.completions.create(**kwargs)
    elapsed = time.perf_counter() - t0
    text = (resp.choices[0].message.content or "").strip()
    usage = getattr(resp, "usage", None)
    return text, _usage_meta(settings, model, temperature, seed, elapsed, usage)


def _amd_inference_completion(
    settings: LLMSettings,
    messages: list[dict[str, Any]],
    *,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int | None,
) -> tuple[str, dict[str, Any]]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        payload["seed"] = seed

    headers = {"Content-Type": "application/json"}
    if settings.api_key:
        headers["Authorization"] = f"Bearer {settings.api_key}"

    url = _join_base_path(settings.base_url, "inference")
    t0 = time.perf_counter()
    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    elapsed = time.perf_counter() - t0
    text = _extract_text(data)
    usage = data.get("usage") if isinstance(data, dict) else None
    return text, _usage_meta(settings, model, temperature, seed, elapsed, usage)


def _join_base_path(base_url: str, path: str) -> str:
    base = base_url.rstrip("/") + "/"
    if base.endswith("/v1/"):
        return urljoin(base, path)
    return urljoin(base, f"v1/{path}")


def _extract_text(data: Any) -> str:
    if not isinstance(data, dict):
        return str(data).strip()

    choices = data.get("choices")
    if isinstance(choices, list) and choices:
        msg = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(msg, dict) and msg.get("content") is not None:
            return _content_to_text(msg["content"])
        if isinstance(choices[0], dict) and choices[0].get("text") is not None:
            return str(choices[0]["text"]).strip()

    for key in ("content", "output", "response", "generated_text", "text"):
        if data.get(key) is not None:
            return _content_to_text(data[key])

    message = data.get("message")
    if isinstance(message, dict) and message.get("content") is not None:
        return _content_to_text(message["content"])

    return str(data).strip()


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                chunks.append(str(item["text"]))
            elif isinstance(item, str):
                chunks.append(item)
        return "\n".join(chunks).strip()
    return str(content).strip()


def _usage_meta(
    settings: LLMSettings,
    model: str,
    temperature: float,
    seed: int | None,
    elapsed: float,
    usage: Any,
) -> dict[str, Any]:
    def _get(name: str) -> Any:
        if isinstance(usage, dict):
            return usage.get(name)
        return getattr(usage, name, None)

    return {
        "provider": settings.provider,
        "api_style": settings.api_style,
        "model": model,
        "temperature": temperature,
        "seed": seed,
        "elapsed_s": round(elapsed, 3),
        "prompt_tokens": _get("prompt_tokens"),
        "completion_tokens": _get("completion_tokens"),
        "total_tokens": _get("total_tokens"),
    }

