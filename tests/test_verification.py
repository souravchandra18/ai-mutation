"""Tests for the citation-grounding verifier.

Pure offline — no network, no LLM. These are the regression tests for
the core methodological contribution of the project.
"""
from __future__ import annotations

from src.verification import (
    build_evidence_index,
    redact_unverified,
    verify_text,
)


SAMPLE_EVIDENCE = {
    "query": {"gene": "BRAF", "protein_change": "V600E", "hgvs": "NM_004333.6:c.1799T>A"},
    "gene": {"symbol": "BRAF", "ensembl": {"gene": "ENSG00000157764"}},
    "variant": {"_id": "chr7:g.140453136A>T", "gnomad_exome": {"af": 0.0}},
    "clinvar": {"rcv": [{"clinical_significance": "Pathogenic"}]},
    "opentargets": {"known_drugs": [{"drug": "Vemurafenib"}]},
    "civic": {
        "variants": [
            {"name": "V600E", "evidence_items": [
                {"id": 123, "pmid": 12345678},
                {"id": 456, "pmid": 23456789},
            ]}
        ]
    },
    "pubmed": {"hits": [{"pmid": 12345678}, {"pmid": 99999999}]},
    "structure": {"uniprot_id": "P15056", "found": True},
}


def test_index_collects_pmids():
    idx = build_evidence_index(SAMPLE_EVIDENCE)
    assert "12345678" in idx.pmids
    assert "23456789" in idx.pmids
    assert "99999999" in idx.pmids
    assert "EID123" in idx.civic_ids
    assert "P15056" in idx.uniprot_ids
    assert idx.opentargets_present
    assert idx.clinvar_present


def test_grounded_citation():
    idx = build_evidence_index(SAMPLE_EVIDENCE)
    text = "BRAF V600E is pathogenic [ClinVar] and PMID-supported [PubMed:12345678]."
    rep = verify_text(text, idx)
    assert rep.total == 2
    assert rep.grounded == 2
    assert rep.citation_grounding_score == 1.0
    assert rep.hallucination_rate == 0.0


def test_fabricated_pmid_is_caught():
    idx = build_evidence_index(SAMPLE_EVIDENCE)
    text = "See [PubMed:11111111] for details."
    rep = verify_text(text, idx)
    assert rep.total == 1
    assert rep.grounded == 0
    assert rep.hallucination_rate == 1.0
    assert "PubMed:11111111" in rep.fabricated_ids


def test_off_context_tag_is_caught():
    # Remove the opentargets block so a bare [OpenTargets] tag is off-context.
    ev = {**SAMPLE_EVIDENCE, "opentargets": {}}
    idx = build_evidence_index(ev)
    text = "Drug X is approved [OpenTargets]."
    rep = verify_text(text, idx)
    assert rep.total == 1
    assert rep.citations[0].status == "off_context"
    assert "OpenTargets" in rep.off_context_tags


def test_redaction_rewrites_unverified():
    idx = build_evidence_index(SAMPLE_EVIDENCE)
    text = "Real [PubMed:12345678]. Fake [PubMed:11111111]."
    rep = verify_text(text, idx)
    redacted = redact_unverified(rep)
    assert "[PubMed:12345678]" in redacted
    assert "[UNVERIFIED:PubMed:11111111]" in redacted
    assert "[PubMed:11111111]" not in redacted


def test_unknown_source_classified():
    idx = build_evidence_index(SAMPLE_EVIDENCE)
    text = "From [SomeFakeDB] we see things."
    rep = verify_text(text, idx)
    assert rep.citations[0].status == "unknown_source"


def test_civic_id_grounding():
    idx = build_evidence_index(SAMPLE_EVIDENCE)
    text = "Per [CIViC:EID123] vemurafenib helps."
    rep = verify_text(text, idx)
    assert rep.grounded == 1
    rep2 = verify_text("Per [CIViC:EID999] something.", idx)
    assert rep2.fabricated_ids == ["CIViC:EID999"]


def test_empty_text_has_perfect_cgs():
    idx = build_evidence_index(SAMPLE_EVIDENCE)
    rep = verify_text("No citations here.", idx)
    assert rep.total == 0
    assert rep.citation_grounding_score == 1.0
    assert rep.hallucination_rate == 0.0
