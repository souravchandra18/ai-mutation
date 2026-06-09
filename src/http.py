"""Shared HTTP client with retries and a small in-memory cache."""
from __future__ import annotations

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

_DEFAULT_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
_client: httpx.Client | None = None
_cache: dict[tuple, object] = {}


def get_client() -> httpx.Client:
    global _client
    if _client is None:
        _client = httpx.Client(
            timeout=_DEFAULT_TIMEOUT,
            headers={"User-Agent": "ai-mutation/0.1 (research)"},
            follow_redirects=True,
        )
    return _client


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type((httpx.HTTPError,)),
)
def get_json(url: str, params: dict | None = None, headers: dict | None = None) -> dict:
    key = ("GET", url, tuple(sorted((params or {}).items())))
    if key in _cache:
        return _cache[key]  # type: ignore[return-value]
    r = get_client().get(url, params=params, headers=headers)
    r.raise_for_status()
    data = r.json()
    _cache[key] = data
    return data


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type((httpx.HTTPError,)),
)
def post_json(url: str, json: dict, headers: dict | None = None) -> dict:
    r = get_client().post(url, json=json, headers=headers)
    r.raise_for_status()
    return r.json()


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type((httpx.HTTPError,)),
)
def get_text(url: str, params: dict | None = None) -> str:
    r = get_client().get(url, params=params)
    r.raise_for_status()
    return r.text
