"""LLM reasoning chain: mutation → mechanism → therapy, grounded in evidence."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .evidence import Evidence
from .llm_client import chat_completion
from .llm_config import get_llm_settings, is_vision_model
from .mutation import MutationQuery
from .verification import (
    GroundingReport,
    aggregate_reports,
    build_evidence_index,
    redact_unverified,
    verify_text,
)

SYSTEM_PROMPT = """You are a careful biomedical reasoning assistant.

You are given STRUCTURED EVIDENCE pulled from public biomedical databases
(MyGene.info, MyVariant.info, ClinVar, Open Targets, CIViC, PubMed,
UniProt + AlphaFold) and from in-house specialist models (ESM-2 protein
language model, BiomedCLIP biomedical image encoder, Whisper speech
transcription). Reason only from this evidence and well-established
background biology. When the evidence is missing or weak, say so explicitly
— do not fabricate.

The evidence includes a `variant_class` field — one of: missense, nonsense,
inframe_deletion, inframe_insertion, inframe_duplication, frameshift,
exon_skip, fusion, amplification, cnv_deletion, hgvs, dbsnp, gene_only.
Tailor your interpretation to the class (e.g. nonsense / frameshift →
likely loss-of-function; amplification → likely gain-of-function via
over-expression; fusion → consider both partners and the chimeric
protein's altered activity / localisation).

If an `evidence.structure` block is provided, use UniProt domain context
(the affected domain, nearby active / binding sites, AlphaFold-predicted
fold) when describing the molecular mechanism. If `evidence.esm2` is
provided, treat its `delta_pll` as a zero-shot variant-effect score from
the ESM-2 protein language model (strongly negative → disrupting; near
zero → tolerated; strongly positive → unusual / context-dependent), and
cite it as `[ESM2]`. If `evidence.imaging` is provided it contains
findings from a biomedical image encoder — cite as `[BiomedCLIP]` and
use it for tissue / disease context only when it is clearly relevant to
the variant under analysis. If `evidence.speech` is provided, treat its
`transcript` as additional free-text instruction from the user.

For every clinical or mechanistic claim, cite the supporting source inline
using bracketed tags like [ClinVar], [OpenTargets], [CIViC],
[PubMed:PMID], [MyVariant], [MyGene], [UniProt], [AlphaFold], [ESM2],
or [BiomedCLIP]. Cite drugs by name and include their development phase
if known.

Output must be production-grade and concise. Avoid speculation about
clinical decision-making for individual patients — this is a research aid,
not medical advice.
"""

STAGE_PROMPTS = {
    "mutation_summary": (
        "STAGE 1 — Mutation Summary.\n"
        "Summarise what this mutation is: gene(s) involved, variant class "
        "(use the `variant_class` field), protein/coding change if "
        "applicable, predicted functional impact (gain-of-function vs "
        "loss-of-function, with reasoning anchored in variant class and "
        "in-silico scores such as REVEL / AlphaMissense / SIFT / PolyPhen-2 "
        "/ CADD, plus the ESM-2 `delta_pll` from `evidence.esm2` if present), "
        "population frequency from gnomAD if available, and ClinVar clinical "
        "significance. 4–8 bullet points."
    ),
    "mechanism": (
        "STAGE 2 — Molecular & Cellular Mechanism.\n"
        "Explain the biological mechanism by which this mutation drives "
        "disease: affected protein domain, downstream pathway(s) (e.g. "
        "MAPK, PI3K/AKT, p53, DNA repair), cellular consequences, and "
        "tissue / disease context. For fusions, address both partners and "
        "the chimeric product. For CNVs, address dosage effects. Reference "
        "the mutation summary from Stage 1. Keep to ~150–250 words."
    ),
    "therapy": (
        "STAGE 3 — Therapeutic Implications.\n"
        "From the evidence, list candidate therapeutic strategies:\n"
        "  • Approved or investigational drugs that target this gene / "
        "pathway (use Open Targets `known_drugs` and any CIViC variants).\n"
        "  • Mechanism-based rationale linking each drug to the mutation.\n"
        "  • Resistance mechanisms or contraindications if mentioned in "
        "evidence.\n"
        "  • Notable clinical trials or biomarkers from PubMed titles.\n"
        "Present as a Markdown table with columns: Drug | Mechanism | "
        "Phase/Status | Rationale | Citation. Follow with a short "
        "'Caveats' paragraph."
    ),
}


@dataclass
class ReasoningResult:
    mutation_summary: str
    mechanism: str
    therapy: str
    # ---- research-mode metadata (populated when deterministic=True or
    # verify=True); empty dicts in legacy code paths so existing callers
    # keep working.
    grounding: dict[str, Any] = field(default_factory=dict)
    redacted: dict[str, str] = field(default_factory=dict)
    run: dict[str, Any] = field(default_factory=dict)

    def to_markdown(self, label: str) -> str:
        return (
            f"# Mutation → Mechanism → Therapy: {label}\n\n"
            f"## 1. Mutation Summary\n{self.mutation_summary}\n\n"
            f"## 2. Molecular Mechanism\n{self.mechanism}\n\n"
            f"## 3. Therapeutic Implications\n{self.therapy}\n"
        )


def _chat(messages: list[dict], temperature: float = 0.2, max_tokens: int = 900,
          model: str | None = None, seed: int | None = None) -> tuple[str, dict[str, Any]]:
    """Send a chat completion. Returns (text, usage_metadata).

    `seed` and `temperature=0` are forwarded so research-mode runs are
    reproducible to the extent the upstream endpoint honours them.
    """
    settings = get_llm_settings()
    chosen = model or settings.model
    return chat_completion(
        settings,
        messages=messages,
        temperature=temperature,
        top_p=1.0 if temperature == 0 else 0.9,
        max_tokens=max_tokens,
        model=chosen,
        seed=seed,
    )


def _prompt_hash(messages: list[dict]) -> str:
    """Stable hash of the full message list for reproducibility logging."""
    blob = json.dumps(messages, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


NO_RAG_NOTICE = (
    "BASELINE MODE: no structured evidence has been retrieved. Answer from "
    "parametric knowledge only. You must still cite sources using the "
    "bracketed tags, but be aware these citations cannot be grounded."
)


def reason(
    mq: MutationQuery,
    ev: Evidence,
    model: str | None = None,
    *,
    deterministic: bool = False,
    verify: bool = False,
    redact: bool = False,
    no_rag: bool = False,
    seed: int | None = None,
) -> ReasoningResult:
    """Three-stage variant reasoning.

    Parameters
    ----------
    deterministic : bool
        If True, sets temperature=0 and forwards a fixed seed so the run
        is reproducible (subject to upstream provider support). Required
        for benchmark evaluation.
    verify : bool
        If True, run the citation-grounding verifier on every stage and
        populate `ReasoningResult.grounding`.
    redact : bool
        If True (and verify is True), rewrite fabricated and off-context
        citations to `[UNVERIFIED:…]` in the returned strings. The
        un-redacted originals are retained under `ReasoningResult.redacted`.
    no_rag : bool
        Baseline mode: send the LLM only the variant label, **not** the
        retrieved evidence. Used to measure the lift contributed by the
        retrieval layer.
    seed : int | None
        Seed forwarded to the LLM; defaults to 7 in deterministic mode.
    """
    temperature = 0.0 if deterministic else 0.2
    if deterministic and seed is None:
        seed = 7
    settings = get_llm_settings()
    chosen_model = model or settings.model

    ev_dict = ev.to_dict()
    plot_b64: str | None = None
    structure_3d_b64: str | None = None
    if isinstance(ev_dict.get("structure"), dict):
        plot_b64 = ev_dict["structure"].pop("domain_plot_png_b64", None)
        structure_3d_b64 = ev_dict["structure"].pop("structure_3d_png_b64", None)

    has_image = bool(plot_b64) or bool(structure_3d_b64)
    use_vision = has_image and is_vision_model(chosen_model) and not no_rag

    # Optional speech-derived free-text instruction.
    speech_block = ev_dict.get("speech") or {}
    speech_transcript = (
        speech_block.get("transcript") if isinstance(speech_block, dict) else None
    )

    if no_rag:
        base_user = (
            f"Mutation under analysis: **{mq.label}**\n\n{NO_RAG_NOTICE}"
        )
    else:
        evidence_json = json.dumps(ev_dict, indent=2, default=str)
        base_user = (
            f"Mutation under analysis: **{mq.label}**\n\n"
            f"STRUCTURED EVIDENCE (JSON):\n```json\n{evidence_json}\n```"
        )
        if speech_transcript:
            base_user += (
                "\n\nADDITIONAL FREE-TEXT INSTRUCTION (from clinician voice note, "
                f"transcribed by Whisper):\n> {speech_transcript}"
            )

    base_messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": base_user},
    ]

    run_meta: dict[str, Any] = {
        **settings.safe_dict(),
        "model": chosen_model,
        "deterministic": deterministic,
        "no_rag": no_rag,
        "vision": use_vision,
        "seed": seed,
        "prompt_hash": _prompt_hash(base_messages),
        "stages": {},
    }

    # Stage 1
    msgs = base_messages + [{"role": "user", "content": STAGE_PROMPTS["mutation_summary"]}]
    stage1, m1 = _chat(msgs, temperature=temperature, max_tokens=600,
                       model=chosen_model, seed=seed)
    run_meta["stages"]["mutation_summary"] = m1

    # Stage 2
    if use_vision:
        mech_text = (
            STAGE_PROMPTS["mechanism"]
            + "\n\nThe attached image(s) come from the structural layer for "
            f"{mq.gene or 'this protein'}. The first is the UniProt domain "
            "map (mutated residue marked as a red lollipop). "
            + ("The second is a 3-D backbone trace from the AlphaFold "
               "predicted structure with the mutated residue as a red sphere "
               "and active / binding sites as orange spheres. " if structure_3d_b64 else "")
            + "Use them to anchor your description of the affected domain, "
            "neighbouring active / binding sites, and the structural "
            "rationale for gain- vs loss-of-function. Cite as [UniProt] / "
            "[AlphaFold]."
        )
        mech_user_content: Any = [{"type": "text", "text": mech_text}]
        if plot_b64:
            mech_user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{plot_b64}"},
            })
        if structure_3d_b64:
            mech_user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{structure_3d_b64}"},
            })
    else:
        mech_user_content = STAGE_PROMPTS["mechanism"]

    msgs = base_messages + [
        {"role": "assistant", "content": stage1},
        {"role": "user", "content": mech_user_content},
    ]
    stage2, m2 = _chat(msgs, temperature=temperature, max_tokens=700,
                       model=chosen_model, seed=seed)
    run_meta["stages"]["mechanism"] = m2

    # Stage 3
    msgs = base_messages + [
        {"role": "assistant", "content": f"Stage 1:\n{stage1}\n\nStage 2:\n{stage2}"},
        {"role": "user", "content": STAGE_PROMPTS["therapy"]},
    ]
    stage3, m3 = _chat(msgs, temperature=temperature, max_tokens=900,
                       model=chosen_model, seed=seed)
    run_meta["stages"]["therapy"] = m3

    result = ReasoningResult(
        mutation_summary=stage1, mechanism=stage2, therapy=stage3, run=run_meta
    )

    if verify:
        idx = build_evidence_index(ev_dict)
        reports: dict[str, GroundingReport] = {
            "mutation_summary": verify_text(stage1, idx),
            "mechanism": verify_text(stage2, idx),
            "therapy": verify_text(stage3, idx),
        }
        result.grounding = {
            "per_stage": {k: r.to_dict() for k, r in reports.items()},
            "aggregate": aggregate_reports(reports.values()),
        }
        if redact:
            result.redacted = {
                "mutation_summary": result.mutation_summary,
                "mechanism": result.mechanism,
                "therapy": result.therapy,
            }
            result.mutation_summary = redact_unverified(reports["mutation_summary"])
            result.mechanism = redact_unverified(reports["mechanism"])
            result.therapy = redact_unverified(reports["therapy"])

    return result
