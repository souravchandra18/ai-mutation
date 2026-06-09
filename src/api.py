"""FastAPI backend exposing the mutation→mechanism→therapy engine.

Run:
    uvicorn src.api:app --reload --port 8000
"""
from __future__ import annotations

import sys
import shutil
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .evidence import gather
from .reasoning import reason

app = FastAPI(
    title="Mutation → Mechanism → Therapy API",
    version="0.2.0",
    description=(
        "Reasons from a genomic mutation to molecular mechanism and "
        "therapeutic implications using public biomedical sources, "
        "specialist models (ESM-2, BiomedCLIP, Whisper), and an "
        "AMD/OpenAI-compatible LLM endpoint. Research use only."
    ),
)

# Permissive CORS so the Streamlit frontend (and local tools) can call us.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class MutationRequest(BaseModel):
    mutation: str = Field(
        ...,
        min_length=1,
        max_length=400,
        examples=["BRAF V600E", "TP53 R175H", "rs113488022", "BRAF V600E;K601E"],
        description='Gene + protein change, HGVS, rsID, fusion, or compound (`;`-separated).',
    )
    model: str | None = Field(
        default=None,
        max_length=200,
        description="Optional model id (overrides AI_MODEL / AMD_MODEL env vars).",
    )


class EvidenceResponse(BaseModel):
    mutation: dict[str, Any]
    evidence: dict[str, Any]


class ReasoningPayload(BaseModel):
    mutation_summary: str
    mechanism: str
    therapy: str


class AnalyzeResponse(BaseModel):
    mutation: dict[str, Any]
    evidence: dict[str, Any]
    reasoning: ReasoningPayload
    grounding: dict[str, Any] = Field(default_factory=dict)
    run: dict[str, Any] = Field(default_factory=dict)


def _optional_dependency_status() -> dict[str, str]:
    status: dict[str, str] = {}
    for module_name in ("torch", "transformers", "PIL", "open_clip", "torchaudio"):
        try:
            module = __import__(module_name)
            status[module_name] = str(getattr(module, "__version__", "installed"))
        except Exception as e:  # noqa: BLE001
            status[module_name] = f"unavailable: {e}"
    try:
        import torch  # type: ignore

        status["torch.cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            status["torch.cuda_device"] = torch.cuda.get_device_name(0)
    except Exception as e:  # noqa: BLE001
        status["torch.cuda_available"] = f"unavailable: {e}"
    status["ffmpeg"] = shutil.which("ffmpeg") or "unavailable"
    return status


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "python": sys.executable,
        "optional_dependencies": _optional_dependency_status(),
    }


@app.post("/evidence")
def evidence_endpoint(req: MutationRequest) -> EvidenceResponse:
    try:
        _, ev = gather(req.mutation)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Evidence gathering failed: {e}") from e
    return EvidenceResponse(mutation=ev.query, evidence=ev.to_dict())


@app.post("/analyze")
def analyze_endpoint(req: MutationRequest) -> AnalyzeResponse:
    try:
        mq, ev = gather(req.mutation)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Evidence gathering failed: {e}") from e

    try:
        result = reason(mq, ev, model=req.model, verify=True, redact=True)
    except RuntimeError as e:
        # Missing API key, etc.
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"LLM reasoning failed: {e}") from e

    return AnalyzeResponse(
        mutation=ev.query,
        evidence=ev.to_dict(),
        reasoning=ReasoningPayload(
            mutation_summary=result.mutation_summary,
            mechanism=result.mechanism,
            therapy=result.therapy,
        ),
        grounding=result.grounding,
        run=result.run,
    )


@app.post("/analyze_mm")
async def analyze_multimodal_endpoint(
    mutation: str = Form(..., min_length=1, max_length=400),
    model: str | None = Form(default=None, max_length=200),
    image: UploadFile | None = File(default=None),
    voice: UploadFile | None = File(default=None),
) -> AnalyzeResponse:
    """Multimodal analysis: variant + optional biomedical image + voice note.

    `image` is scored by BiomedCLIP / CLIP and surfaced as `evidence.imaging`.
    `voice` is transcribed by Whisper, the transcript is appended to the
    user prompt, and the speech block is included under `evidence.speech`.
    """
    image_bytes = await image.read() if image is not None else None
    voice_bytes = await voice.read() if voice is not None else None

    try:
        mq, ev = gather(mutation, image=image_bytes, voice=voice_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Evidence gathering failed: {e}") from e

    try:
        result = reason(mq, ev, model=model, verify=True, redact=True)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"LLM reasoning failed: {e}") from e

    result.run["uploads"] = {
        "image_received": image is not None,
        "image_filename": image.filename if image is not None else None,
        "image_bytes": len(image_bytes or b""),
        "voice_received": voice is not None,
        "voice_filename": voice.filename if voice is not None else None,
        "voice_bytes": len(voice_bytes or b""),
    }

    return AnalyzeResponse(
        mutation=ev.query,
        evidence=ev.to_dict(),
        reasoning=ReasoningPayload(
            mutation_summary=result.mutation_summary,
            mechanism=result.mechanism,
            therapy=result.therapy,
        ),
        grounding=result.grounding,
        run=result.run,
    )
