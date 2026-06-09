"""Speech-in modality — Whisper transcription for clinician voice notes.

Uses HuggingFace `transformers` pipeline so it works on PyTorch-ROCm
without the CUDA-only `faster-whisper` build. The first call loads the
model; subsequent calls reuse it. Heavy imports are lazy.

Default model: `openai/whisper-base` (~140 MB) — fast and adequate for
short clinical dictation. Override with `WHISPER_MODEL` for a larger /
domain-specific model (e.g. `openai/whisper-large-v3`).
"""
from __future__ import annotations

import io
import os
from functools import lru_cache
from typing import Any

DEFAULT_MODEL = os.getenv("WHISPER_MODEL", "openai/whisper-base")
ENABLED = os.getenv("WHISPER_ENABLED", "1") != "0"


@lru_cache(maxsize=1)
def _load_pipeline() -> Any | None:
    try:
        import torch  # type: ignore
        from transformers import pipeline  # type: ignore
    except Exception:  # noqa: BLE001
        return None
    try:
        device = 0 if torch.cuda.is_available() else -1
        return pipeline(
            "automatic-speech-recognition",
            model=DEFAULT_MODEL,
            device=device,
        )
    except Exception:  # noqa: BLE001
        return None


def _coerce_input(audio: Any) -> Any:
    """Accept path | bytes | file-like and return a value the pipeline takes.

    `transformers` ASR pipeline accepts a path string or raw bytes directly.
    """
    if isinstance(audio, str):
        return audio
    if isinstance(audio, (bytes, bytearray)):
        return bytes(audio)
    if hasattr(audio, "read"):
        return audio.read()
    raise TypeError(f"Unsupported audio input: {type(audio)}")


def transcribe(audio: Any) -> dict[str, Any]:
    """Transcribe audio to text. Returns a JSON-serialisable dict.

    `audio` may be a filesystem path, raw bytes, or a file-like object.
    """
    if not ENABLED:
        return {"found": False, "reason": "WHISPER_ENABLED=0"}

    asr = _load_pipeline()
    if asr is None:
        return {
            "found": False,
            "reason": (
                "transformers/torch unavailable — install with "
                "`pip install transformers torch torchaudio`."
            ),
        }
    try:
        payload = _coerce_input(audio)
    except TypeError as e:
        return {"found": False, "reason": str(e)}

    try:
        # `return_timestamps=False` keeps output small for short clips.
        result = asr(payload)
        if isinstance(result, dict):
            text = (result.get("text") or "").strip()
        else:  # list of chunks
            text = " ".join(c.get("text", "") for c in result).strip()
    except Exception as e:  # noqa: BLE001
        return {"found": False, "reason": f"ASR failed: {e}"}

    return {
        "found": True,
        "model": DEFAULT_MODEL,
        "transcript": text,
    }
