"""Citation-grounding verifier.

This module implements the core methodological contribution of the
project: a post-hoc verifier that audits every citation an LLM emits in
its variant-interpretation output and flags fabricated, unsupported, or
out-of-context references.

Two failure modes are caught:

1. **Fabricated identifier** — the LLM cites a PubMed PMID, ClinVar ID,
   CIViC evidence ID, or Open Targets / UniProt accession that does not
   exist in the structured evidence that was provided as context. This
   is the dominant biomedical-LLM hallucination class (Ji et al., 2023;
   Singhal et al., 2023).

2. **Off-context source tag** — the LLM cites a database tag (e.g.
   `[ClinVar]`, `[CIViC]`) for which the retrieved evidence block was
   empty. The model is asserting support that the retrieval layer never
   supplied.

For each piece of generated text we return a `GroundingReport` carrying:

* per-citation outcome (`grounded` / `fabricated` / `off_context`)
* an aggregate **citation-grounding score** (CGS) in [0, 1]
* lists of fabricated IDs and unsupported source tags
* a redacted version of the text with `[UNVERIFIED:<tag>]` substituted
  for every fabricated citation, suitable for downstream display

The verifier is **deterministic** — it does no network calls and no LLM
calls — and therefore safe to run inside CI and unit tests.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

# ---------------------------------------------------------------------------
# Citation extraction
# ---------------------------------------------------------------------------

# Matches bracketed citations the system prompt asks the model to emit, e.g.:
#   [PubMed:12345678]   [ClinVar]   [CIViC:EID123]   [OpenTargets]
#   [UniProt:P15056]    [AlphaFold] [MyVariant]      [MyGene]
_CITATION_RE = re.compile(
    r"\[(?P<source>[A-Za-z][A-Za-z0-9 _-]*?)"
    r"(?::(?P<ident>[A-Za-z0-9._:-]+))?"
    r"\]"
)

# Sources we recognise. Anything else is reported as `unknown_source`.
_KNOWN_SOURCES = {
    "pubmed", "clinvar", "civic", "opentargets", "open_targets",
    "mygene", "myvariant", "uniprot", "alphafold", "gnomad", "ensembl",
    "mutalyzer", "esm2", "biomedclip",
}

# Sources for which we require the cited identifier to appear in the
# retrieved evidence (strict ID-grounding). For tag-only citations like
# `[ClinVar]` we instead require that the corresponding evidence block
# is non-empty (soft grounding).
_ID_GROUNDED_SOURCES = {"pubmed", "civic", "uniprot", "alphafold"}


@dataclass
class CitationOutcome:
    source: str
    identifier: str | None
    status: str  # one of: grounded | fabricated | off_context | unknown_source
    span: tuple[int, int]  # character offsets in the input text

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "identifier": self.identifier,
            "status": self.status,
            "span": list(self.span),
        }


@dataclass
class GroundingReport:
    """Aggregate outcome of verifying one LLM-generated string."""

    text: str
    citations: list[CitationOutcome] = field(default_factory=list)
    fabricated_ids: list[str] = field(default_factory=list)
    off_context_tags: list[str] = field(default_factory=list)
    unknown_sources: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.citations)

    @property
    def grounded(self) -> int:
        return sum(1 for c in self.citations if c.status == "grounded")

    @property
    def citation_grounding_score(self) -> float:
        """Fraction of emitted citations that are properly grounded in the
        retrieved evidence. Returns 1.0 when no citations were emitted
        (vacuously grounded — caller may want to penalise empty output
        separately)."""
        if not self.citations:
            return 1.0
        return self.grounded / self.total

    @property
    def hallucination_rate(self) -> float:
        """Fraction of citations that are fabricated identifiers."""
        if not self.citations:
            return 0.0
        fab = sum(1 for c in self.citations if c.status == "fabricated")
        return fab / self.total

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_citations": self.total,
            "grounded": self.grounded,
            "citation_grounding_score": round(self.citation_grounding_score, 4),
            "hallucination_rate": round(self.hallucination_rate, 4),
            "fabricated_ids": self.fabricated_ids,
            "off_context_tags": self.off_context_tags,
            "unknown_sources": self.unknown_sources,
            "citations": [c.to_dict() for c in self.citations],
        }


# ---------------------------------------------------------------------------
# Evidence-index extraction
# ---------------------------------------------------------------------------


@dataclass
class EvidenceIndex:
    """Set-of-identifiers view of the structured evidence dict.

    Built once per variant and reused across all three reasoning stages.
    """

    pmids: set[str] = field(default_factory=set)
    civic_ids: set[str] = field(default_factory=set)
    uniprot_ids: set[str] = field(default_factory=set)
    alphafold_ids: set[str] = field(default_factory=set)
    clinvar_present: bool = False
    opentargets_present: bool = False
    mygene_present: bool = False
    myvariant_present: bool = False
    gnomad_present: bool = False
    ensembl_present: bool = False
    mutalyzer_present: bool = False
    esm2_present: bool = False
    biomedclip_present: bool = False

    def has_source(self, source: str) -> bool:
        s = source.lower().replace("-", "_")
        if s in ("open_targets", "opentargets"):
            return self.opentargets_present
        return getattr(self, f"{s}_present", False) or bool(
            self._ids_for(s)
        )

    def _ids_for(self, source: str) -> set[str]:
        s = source.lower().replace("-", "_")
        if s == "pubmed":
            return self.pmids
        if s == "civic":
            return self.civic_ids
        if s == "uniprot":
            return self.uniprot_ids
        if s == "alphafold":
            return self.alphafold_ids
        return set()


def _walk_for_pmids(node: Any, out: set[str]) -> None:
    """Recursively harvest anything that looks like a PubMed ID."""
    if isinstance(node, dict):
        for k, v in node.items():
            kl = str(k).lower()
            if kl in ("pmid", "pubmed_id", "pmcid") and v:
                out.add(str(v).lstrip("PMC").lstrip("PMID:").strip())
            else:
                _walk_for_pmids(v, out)
    elif isinstance(node, list):
        for item in node:
            _walk_for_pmids(item, out)
    elif isinstance(node, (str, int)):
        s = str(node)
        # Bare 6-9 digit numerics inside curated lists are very likely PMIDs.
        if s.isdigit() and 5 <= len(s) <= 9:
            out.add(s)


def build_evidence_index(evidence: dict[str, Any]) -> EvidenceIndex:
    """Construct the index from an `Evidence.to_dict()` payload."""
    idx = EvidenceIndex()

    pubmed = evidence.get("pubmed") or {}
    if pubmed:
        _walk_for_pmids(pubmed, idx.pmids)
    # PubMed IDs sometimes ride along inside CIViC or ClinVar payloads.
    _walk_for_pmids(evidence.get("civic"), idx.pmids)
    _walk_for_pmids(evidence.get("clinvar"), idx.pmids)

    civic = evidence.get("civic") or {}
    for v in (civic.get("variants") or []):
        for ev in (v.get("evidence_items") or v.get("evidenceItems") or []):
            eid = ev.get("id") or ev.get("name")
            if eid:
                idx.civic_ids.add(f"EID{eid}" if str(eid).isdigit() else str(eid))

    struct = evidence.get("structure") or {}
    if struct.get("uniprot_id"):
        idx.uniprot_ids.add(str(struct["uniprot_id"]))
        idx.alphafold_ids.add(str(struct["uniprot_id"]))

    idx.clinvar_present = bool(evidence.get("clinvar") or {})
    ot = evidence.get("opentargets") or {}
    idx.opentargets_present = bool(ot)
    idx.mygene_present = bool(evidence.get("gene"))
    idx.myvariant_present = bool(evidence.get("variant"))
    # gnomAD / Ensembl arrive nested in MyVariant payloads.
    mv = evidence.get("variant") or {}
    idx.gnomad_present = "gnomad_exome" in mv or "gnomad_genome" in mv
    idx.ensembl_present = bool(mv.get("ensembl")) or bool(
        (evidence.get("gene") or {}).get("ensembl")
    )
    idx.mutalyzer_present = bool((evidence.get("query") or {}).get("hgvs"))

    esm = evidence.get("esm2") or {}
    idx.esm2_present = bool(esm and esm.get("found"))
    img = evidence.get("imaging") or {}
    idx.biomedclip_present = bool(img and img.get("found"))

    return idx


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


def _normalise_ident(source: str, ident: str) -> str:
    s = source.lower()
    if s == "pubmed":
        return ident.lstrip("PMID:").lstrip("PMC").strip()
    if s == "civic":
        # accept both "EID123" and "123"
        return ident if ident.upper().startswith("EID") else f"EID{ident}"
    return ident.strip()


def verify_text(text: str, idx: EvidenceIndex) -> GroundingReport:
    """Audit one block of LLM output against the supplied evidence index."""
    report = GroundingReport(text=text)

    for m in _CITATION_RE.finditer(text):
        raw_source = (m.group("source") or "").strip()
        ident = (m.group("ident") or "").strip() or None
        source_key = raw_source.lower().replace("-", "_").replace(" ", "_")

        if source_key not in _KNOWN_SOURCES:
            report.citations.append(
                CitationOutcome(raw_source, ident, "unknown_source", m.span())
            )
            report.unknown_sources.append(raw_source)
            continue

        if ident and source_key in _ID_GROUNDED_SOURCES:
            normalised = _normalise_ident(source_key, ident)
            present = normalised in idx._ids_for(source_key)
            if present:
                report.citations.append(
                    CitationOutcome(raw_source, ident, "grounded", m.span())
                )
            else:
                report.citations.append(
                    CitationOutcome(raw_source, ident, "fabricated", m.span())
                )
                report.fabricated_ids.append(f"{raw_source}:{ident}")
        else:
            # Tag-only citation — require the source block to be non-empty.
            if idx.has_source(source_key):
                report.citations.append(
                    CitationOutcome(raw_source, ident, "grounded", m.span())
                )
            else:
                report.citations.append(
                    CitationOutcome(raw_source, ident, "off_context", m.span())
                )
                report.off_context_tags.append(raw_source)

    return report


def redact_unverified(report: GroundingReport) -> str:
    """Return the original text with fabricated and off-context citations
    rewritten as `[UNVERIFIED:<tag>]` so they cannot be silently trusted
    by a downstream reader."""
    text = report.text
    # Walk backwards so spans stay valid as we mutate the string.
    for c in sorted(report.citations, key=lambda x: x.span[0], reverse=True):
        if c.status in ("fabricated", "off_context"):
            start, end = c.span
            tag = c.source if not c.identifier else f"{c.source}:{c.identifier}"
            text = f"{text[:start]}[UNVERIFIED:{tag}]{text[end:]}"
    return text


def aggregate_reports(reports: Iterable[GroundingReport]) -> dict[str, Any]:
    """Aggregate per-stage reports into a single per-variant summary."""
    reports = list(reports)
    total = sum(r.total for r in reports)
    grounded = sum(r.grounded for r in reports)
    fab = sum(1 for r in reports for c in r.citations if c.status == "fabricated")
    off = sum(1 for r in reports for c in r.citations if c.status == "off_context")
    unk = sum(1 for r in reports for c in r.citations if c.status == "unknown_source")
    return {
        "total_citations": total,
        "grounded": grounded,
        "fabricated": fab,
        "off_context": off,
        "unknown_source": unk,
        "citation_grounding_score": round(grounded / total, 4) if total else 1.0,
        "hallucination_rate": round(fab / total, 4) if total else 0.0,
    }
