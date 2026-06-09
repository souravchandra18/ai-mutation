"""Evidence aggregator: pulls all data-source results for a mutation.

All independent network calls run concurrently in a thread pool — the
underlying HTTP client is sync (httpx + tenacity) so threads give us
parallelism without rewriting the data-source layer as async.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

from . import data_sources as ds
from . import imaging as ds_imaging
from . import protein_lm as ds_esm
from . import structure as ds_struct
from .mutation import MutationQuery, VariantClass, parse_mutation

_MAX_WORKERS = int(os.getenv("EVIDENCE_MAX_WORKERS", "8"))


@dataclass
class Evidence:
    query: dict[str, Any]
    gene: dict[str, Any] = field(default_factory=dict)
    variant: dict[str, Any] = field(default_factory=dict)
    clinvar: dict[str, Any] = field(default_factory=dict)
    opentargets: dict[str, Any] = field(default_factory=dict)
    civic: dict[str, Any] = field(default_factory=dict)
    pubmed: dict[str, Any] = field(default_factory=dict)
    structure: dict[str, Any] = field(default_factory=dict)
    esm2: dict[str, Any] = field(default_factory=dict)
    imaging: dict[str, Any] = field(default_factory=dict)
    speech: dict[str, Any] = field(default_factory=dict)
    fusion_partner_evidence: dict[str, Any] = field(default_factory=dict)
    # Populated only for compound queries (e.g. "BRAF V600E;K601E").
    compound_evidence: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _query_dict(mq: MutationQuery) -> dict[str, Any]:
    return {
        "raw": mq.raw,
        "variant_class": mq.variant_class,
        "gene": mq.gene,
        "protein_change": mq.protein_change,
        "hgvs": mq.hgvs,
        "rsid": mq.rsid,
        "fusion_partners": list(mq.fusion_partners) if mq.fusion_partners else None,
        "exon": mq.exon,
        "label": mq.label,
        "compound_parts": (
            [_query_dict(p) for p in mq.compound_parts] if mq.compound_parts else None
        ),
    }


def _canonicalise_in_place(mq: MutationQuery) -> None:
    """Resolve aliases (HER2 → ERBB2) on the query and any compound parts."""
    if mq.gene:
        mq.gene = ds.canonicalize_symbol(mq.gene)
    if mq.fusion_partners:
        a, b = mq.fusion_partners
        mq.fusion_partners = (ds.canonicalize_symbol(a), ds.canonicalize_symbol(b))
    if mq.compound_parts:
        for p in mq.compound_parts:
            _canonicalise_in_place(p)


def _gather_single(mq: MutationQuery, pool: ThreadPoolExecutor) -> Evidence:
    """Gather evidence for a non-compound query, running all sources in parallel."""
    ev = Evidence(query=_query_dict(mq))
    primary_gene = mq.gene or (mq.fusion_partners[0] if mq.fusion_partners else None)

    tasks: dict[str, Any] = {}

    def submit(name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        tasks[name] = pool.submit(fn, *args, **kwargs)

    if primary_gene:
        submit("gene", ds.gene_info, primary_gene)
        submit("opentargets", ds.opentargets_for_gene, primary_gene)
        submit("civic", ds.civic_for_gene, primary_gene, mq.protein_change)

    if mq.fusion_partners:
        _, partner = mq.fusion_partners
        submit("fusion_gene", ds.gene_info, partner)
        submit("fusion_ot", ds.opentargets_for_gene, partner)
        submit("fusion_civic", ds.civic_for_gene, partner)

    submit("variant", ds.variant_info, mq)
    submit("clinvar", ds.clinvar_summary, mq)
    submit("pubmed", ds.pubmed_search, mq)
    submit("structure", ds_struct.fetch_structure, mq)
    submit("esm2", ds_esm.score_variant, mq)

    # Collect (each future already exception-safe inside data_sources).
    def _result(key: str, default: Any) -> Any:
        fut = tasks.get(key)
        if fut is None:
            return default
        try:
            return fut.result()
        except Exception as e:  # noqa: BLE001
            return {"found": False, "error": str(e)}

    if primary_gene:
        ev.gene = _result("gene", {})
        ev.opentargets = _result("opentargets", {})
        ev.civic = _result("civic", {})

    if mq.fusion_partners:
        _, partner = mq.fusion_partners
        ev.fusion_partner_evidence = {
            "symbol": partner,
            "gene": _result("fusion_gene", {}),
            "opentargets": _result("fusion_ot", {}),
            "civic": _result("fusion_civic", {}),
        }

    ev.variant = _result("variant", {})
    ev.clinvar = _result("clinvar", {})
    ev.pubmed = _result("pubmed", {})
    ev.structure = _result("structure", {})
    ev.esm2 = _result("esm2", {})
    return ev


def gather(
    mutation_text: str,
    *,
    image: Any = None,
    voice: Any = None,
) -> tuple[MutationQuery, "Evidence"]:
    """Gather all evidence for a mutation, optionally including
    user-supplied biomedical image and/or voice-note inputs.

    Parameters
    ----------
    image : optional path | bytes | file-like | PIL.Image
        Histology / radiology / microscopy image scored by BiomedCLIP.
    voice : optional path | bytes | file-like
        Clinician voice note transcribed by Whisper. The transcript is
        appended to the parsed mutation as additional free-text context.
    """
    mq = parse_mutation(mutation_text)
    _canonicalise_in_place(mq)

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="evidence") as pool:
        # Kick off the optional user-supplied modalities in parallel
        # alongside the database fan-out.
        image_future = pool.submit(ds_imaging.encode_image, image) if image is not None else None
        from . import speech as ds_speech  # local import keeps optional dep lazy
        speech_future = pool.submit(ds_speech.transcribe, voice) if voice is not None else None

        if mq.compound_parts:
            # Build an aggregate Evidence: gene-level info comes from the (shared)
            # primary gene; per-part evidence is exposed via `compound_evidence`.
            primary = mq.compound_parts[0]
            agg = _gather_single(primary, pool)
            agg.query = _query_dict(mq)

            # Gather the remaining parts in parallel as well — each part still
            # parallelises its own sources internally via the same pool.
            part_evidences = [agg]
            for sub in mq.compound_parts[1:]:
                part_evidences.append(_gather_single(sub, pool))

            agg.compound_evidence = [
                {
                    "query": _query_dict(p_mq),
                    "variant": p_ev.variant,
                    "clinvar": p_ev.clinvar,
                    "pubmed": p_ev.pubmed,
                    "civic": p_ev.civic,
                    "esm2": p_ev.esm2,
                }
                for p_mq, p_ev in zip(mq.compound_parts, part_evidences)
            ]
            ev = agg
        else:
            ev = _gather_single(mq, pool)

        if image_future is not None:
            try:
                ev.imaging = image_future.result() or {}
            except Exception as e:  # noqa: BLE001
                ev.imaging = {"found": False, "error": str(e)}
        if speech_future is not None:
            try:
                ev.speech = speech_future.result() or {}
            except Exception as e:  # noqa: BLE001
                ev.speech = {"found": False, "error": str(e)}

        return mq, ev
