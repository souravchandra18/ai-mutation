"""Biomedical image encoder — vision modality for histology / radiology.

Given an uploaded image (H&E patch, radiology slice, microscopy field) we
run a CLIP-style biomedical vision-language model to score the image
against a curated list of clinical labels. The top matches are returned
as JSON findings that the LLM can cite as `[BiomedCLIP]`.

Two backends are supported (auto-detected at first call):
  1. **BiomedCLIP** via `open_clip_torch` — preferred when installed.
     Model: `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`.
  2. **OpenAI CLIP** via `transformers` — fallback (`openai/clip-vit-base-patch32`).
     Generic, but works without `open_clip_torch`.

All heavy imports are lazy. If neither backend is available, callers
receive `{"found": False, "reason": ...}` and the rest of the pipeline
proceeds normally.
"""
from __future__ import annotations

import io
import os
from functools import lru_cache
from typing import Any

DEFAULT_LABELS: tuple[str, ...] = (
    # Pathology — generic
    "histology of normal tissue",
    "histology of inflammatory infiltrate",
    "histology of fibrosis",
    "histology of necrosis",
    "histology of well-differentiated adenocarcinoma",
    "histology of poorly-differentiated carcinoma",
    "histology of small cell carcinoma",
    "histology of squamous cell carcinoma",
    "histology of melanoma",
    "histology of lymphoma",
    "histology of sarcoma",
    # Radiology — generic
    "chest x-ray with no acute findings",
    "chest x-ray with pulmonary nodule",
    "chest x-ray with consolidation suggestive of pneumonia",
    "CT scan with hepatic lesion",
    "CT scan with brain mass",
    "MRI with white matter hyperintensities",
    # Microscopy
    "fluorescence microscopy of cell culture",
    "electron microscopy of subcellular structures",
)

BIOMEDCLIP_MODEL = os.getenv(
    "BIOMEDCLIP_MODEL",
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
)
CLIP_FALLBACK = os.getenv("CLIP_FALLBACK", "openai/clip-vit-base-patch32")
TOP_K = int(os.getenv("IMAGING_TOP_K", "5"))


# ---------------------------------------------------------------------------
# Backend detection (cached after first call)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _load_biomedclip() -> tuple[Any, Any, Any, Any] | dict[str, str]:
    """Try to load BiomedCLIP via open_clip. Returns loaded objects or an error."""
    try:
        import torch  # type: ignore
        import open_clip  # type: ignore
    except Exception as e:  # noqa: BLE001
        return {"error": f"BiomedCLIP import failed: {e}"}
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(BIOMEDCLIP_MODEL)
        tokenizer = open_clip.get_tokenizer(BIOMEDCLIP_MODEL)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()
        return model, preprocess, tokenizer, device
    except Exception as e:  # noqa: BLE001
        return {"error": f"BiomedCLIP load failed: {e}"}


@lru_cache(maxsize=1)
def _load_clip_fallback() -> tuple[Any, Any, Any] | dict[str, str]:
    """Generic OpenAI CLIP via transformers. Returns loaded objects or an error."""
    try:
        import torch  # type: ignore
        from transformers import CLIPModel, CLIPProcessor  # type: ignore
    except Exception as e:  # noqa: BLE001
        return {"error": f"CLIP import failed: {e}"}
    try:
        model = CLIPModel.from_pretrained(CLIP_FALLBACK)
        processor = CLIPProcessor.from_pretrained(CLIP_FALLBACK)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()
        return model, processor, device
    except Exception as e:  # noqa: BLE001
        return {"error": f"CLIP load failed: {e}"}


def _open_image(image: Any) -> Any | None:
    try:
        from PIL import Image  # type: ignore
    except Exception:  # noqa: BLE001
        return None
    try:
        if hasattr(image, "read"):
            return Image.open(image).convert("RGB")
        if isinstance(image, (bytes, bytearray)):
            if not image:
                return None
            return Image.open(io.BytesIO(image)).convert("RGB")
        if isinstance(image, str) and os.path.isfile(image):
            return Image.open(image).convert("RGB")
        return Image.fromarray(image).convert("RGB")  # numpy array
    except Exception:  # noqa: BLE001
        return None


def _encode_with_biomedclip(pil_image: Any, labels: tuple[str, ...]) -> dict[str, Any]:
    loaded = _load_biomedclip()
    if isinstance(loaded, dict):
        return {"found": False, "reason": loaded["error"]}
    model, preprocess, tokenizer, device = loaded

    import torch  # type: ignore

    image = preprocess(pil_image).unsqueeze(0).to(device)
    text = tokenizer(list(labels)).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        sims = (100.0 * image_features @ text_features.T).softmax(dim=-1)[0]

    return _format_topk(labels, sims.tolist(), backend="BiomedCLIP",
                        model_id=BIOMEDCLIP_MODEL, device=device)


def _encode_with_clip(pil_image: Any, labels: tuple[str, ...]) -> dict[str, Any]:
    loaded = _load_clip_fallback()
    if isinstance(loaded, dict):
        return {"found": False, "reason": loaded["error"]}
    model, processor, device = loaded

    import torch  # type: ignore

    inputs = processor(text=list(labels), images=pil_image,
                       return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        out = model(**inputs)
        logits_per_image = out.logits_per_image[0]
        probs = logits_per_image.softmax(dim=-1)

    return _format_topk(labels, probs.tolist(), backend="CLIP",
                        model_id=CLIP_FALLBACK, device=device)


def _format_topk(labels: tuple[str, ...], probs: list[float],
                 *, backend: str, model_id: str, device: str) -> dict[str, Any]:
    paired = sorted(zip(labels, probs), key=lambda kv: kv[1], reverse=True)
    top = paired[:TOP_K]
    return {
        "found": True,
        "backend": backend,
        "model": model_id,
        "device": device,
        "top_findings": [
            {"label": lbl, "score": round(float(p), 4)} for lbl, p in top
        ],
        "summary": "; ".join(f"{lbl} ({p:.2f})" for lbl, p in top[:3]),
    }


def encode_image(image: Any,
                 labels: tuple[str, ...] | list[str] | None = None) -> dict[str, Any]:
    """Score an uploaded biomedical image against a label panel.

    Parameters
    ----------
    image : path | bytes | file-like | PIL.Image | numpy.ndarray
    labels : optional caller-supplied list of label prompts (overrides defaults)
    """
    pil = _open_image(image)
    if pil is None:
        return {"found": False, "reason": "could not open image (Pillow missing or bad input)"}
    label_tuple: tuple[str, ...] = tuple(labels) if labels else DEFAULT_LABELS

    # Prefer BiomedCLIP; fall back to generic CLIP while preserving diagnostics.
    biomed = _encode_with_biomedclip(pil, label_tuple)
    if biomed.get("found"):
        return biomed
    clip = _encode_with_clip(pil, label_tuple)
    if clip.get("found"):
        clip["fallback_reason"] = biomed.get("reason")
        return clip
    return {
        "found": False,
        "reason": "No image encoder available.",
        "biomedclip_error": biomed.get("reason"),
        "clip_error": clip.get("reason"),
    }
