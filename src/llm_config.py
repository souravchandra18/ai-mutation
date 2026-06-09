"""LLM provider configuration.

The application talks to an OpenAI-compatible chat endpoint. On AMD AI
Developer Cloud this is typically a vLLM or AMD AIM deployment running on an
AMD Instinct GPU instance.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

DEFAULT_PROVIDER = "amd"
DEFAULT_BASE_URL = "http://localhost:8090/v1"
DEFAULT_TEXT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_VISION_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

DEFAULT_MODELS = [
    DEFAULT_TEXT_MODEL,
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "mistralai/Mixtral-8x22B-Instruct-v0.1",
    DEFAULT_VISION_MODEL,
    "Qwen/Qwen2-VL-7B-Instruct",
    "OpenGVLab/InternVL3-8B",
]


VISION_KEYWORDS = (
    "vision", "-vl", "-vl-", "vlm", "llava", "pixtral",
    "qwen-vl", "qwen2-vl", "qwen2.5-vl", "internvl", "phi-3.5-vision"
)


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    api_key: str
    base_url: str
    model: str
    vision_model: str
    api_style: str

    @property
    def display_provider(self) -> str:
        return self.provider.upper() if self.provider else "OPENAI-COMPATIBLE"

    def safe_dict(self) -> dict[str, str]:
        return {
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "vision_model": self.vision_model,
            "api_style": self.api_style,
        }


def _env_first(*names: str, default: str = "") -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return default


def get_llm_settings() -> LLMSettings:
    """Read AMD-first LLM configuration with legacy environment fallbacks."""
    provider = _env_first("AI_PROVIDER", "AMD_PROVIDER", default=DEFAULT_PROVIDER).lower()
    api_key = _env_first("AI_API_KEY", "AMD_API_KEY", "NVIDIA_API_KEY")
    base_url = _env_first(
        "AI_BASE_URL",
        "AMD_BASE_URL",
        "NVIDIA_BASE_URL",
        default=DEFAULT_BASE_URL,
    ).rstrip("/")
    model = _env_first(
        "AI_MODEL",
        "AMD_MODEL",
        "NVIDIA_MODEL",
        default=DEFAULT_TEXT_MODEL,
    )
    vision_model = _env_first(
        "AI_VISION_MODEL",
        "AMD_VISION_MODEL",
        "NVIDIA_VISION_MODEL",
        default=DEFAULT_VISION_MODEL,
    )
    api_style = _env_first("AI_API_STYLE", "AMD_API_STYLE", default="openai").lower()
    return LLMSettings(
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        model=model,
        vision_model=vision_model,
        api_style=api_style,
    )


def is_vision_model(model: str | None) -> bool:
    value = (model or "").lower()
    return any(keyword in value for keyword in VISION_KEYWORDS)
