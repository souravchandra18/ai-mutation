---
title: 'ai-mutation: a citation-grounded, multi-modal retrieval-augmented pipeline for variant interpretation'
tags:
  - Python
  - bioinformatics
  - variant interpretation
  - large language models
  - retrieval-augmented generation
  - hallucination mitigation
  - precision oncology
authors:
  - name: <Author One>
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: <Affiliation>, <City>, <Country>
    index: 1
date: 26 May 2026
bibliography: paper.bib
---

# Summary

`ai-mutation` is an open-source Python application that turns a free-text
mutation query (e.g. `BRAF V600E`, `BCR::ABL1`, `MET exon14skip`,
`NM_004333.6:c.1799T>A`) into an evidence-grounded, three-stage
biomedical reasoning report covering (i) the variant summary,
(ii) molecular and cellular mechanism, and (iii) candidate therapeutic
strategies. Evidence is pulled in parallel from MyGene.info,
MyVariant.info, NCBI ClinVar and PubMed, the CIViC GraphQL API, Open
Targets, UniProt, AlphaFold and Mutalyzer 3; a UniProt-derived domain
map is rendered to a PNG and, when a vision-capable language model is
selected, attached to the second-stage prompt as a multi-modal grounding
signal. Every claim emitted by the LLM is post-hoc audited by a
deterministic **citation-grounding verifier** that flags fabricated
PubMed / CIViC / UniProt identifiers and off-context source tags,
yielding a per-output **Citation-Grounding Score (CGS)** and
**hallucination rate**.

# Statement of need

Large language models are increasingly used as front-ends to biomedical
knowledge bases, but they remain prone to fabricating identifiers —
inventing plausible-looking PubMed IDs, drug names, or trial numbers
that do not exist in the retrieved context [@ji2023; @singhal2023].
This is a particular liability in variant interpretation, where a
fabricated citation can be indistinguishable from a real one to a
non-expert reader, and where downstream clinical decisions may be
influenced by the output even though the system is research-use-only.

Existing variant-interpretation tools fall into two camps: (1) curated
knowledge bases (CIViC [@civic2017], OncoKB [@oncokb2017], ClinGen
[@clingen2018]) that return only what humans have already vetted, with
no synthesis across sources; and (2) general LLM front-ends that
synthesise freely but cannot be audited. `ai-mutation` occupies a
middle ground: it performs structured, parallelised retrieval across
nine public sources, asks the LLM to reason only over that retrieved
context with explicit `[Source:ID]` citations, and then *verifies every
citation* against the retrieved evidence index. Citations that fail
verification can either be reported as a metric (for evaluation) or
rewritten as `[UNVERIFIED:…]` in the displayed output (for end-user
safety).

# Methodological contribution

The novel component is the citation-grounding verifier in
`src/verification.py`. Given (a) an LLM-emitted text and (b) the
structured evidence dictionary that was supplied as retrieval context,
the verifier:

1. Builds an `EvidenceIndex` of every PubMed PMID, CIViC evidence ID
   (`EID`), UniProt accession and source-block presence flag in the
   retrieval payload (recursively, so identifiers embedded inside
   nested CIViC / ClinVar records are also captured).
2. Extracts every `[Source]` or `[Source:Identifier]` citation from the
   LLM text using a fixed grammar.
3. Classifies each citation as **grounded**, **fabricated** (identifier
   not in the index), **off-context** (source block was empty), or
   **unknown_source**.
4. Returns a `GroundingReport` containing per-citation outcomes, the
   CGS, and the hallucination rate; optionally produces a redacted
   version of the text with all unverified citations rewritten.

The verifier is deterministic, contains no network calls or LLM calls,
and runs in milliseconds, which makes it usable both as a metric inside
the evaluation harness and as a real-time guard-rail in the UI.

# Architecture

```
        ┌─────────────────────────────────────────────────────────┐
        │  parse → canonicalise → parallel retrieval (9 sources)  │
        └────────────────────────┬────────────────────────────────┘
                                 │ Evidence (JSON)
                                 ▼
         ┌─────────── 3-stage LLM reasoning (deterministic mode) ─┐
         │   stage 1: mutation summary                             │
         │   stage 2: mechanism  ─── optional multi-modal: ──────►│  vision LLM
         │                              domain-map PNG            │
         │   stage 3: therapy                                      │
         └────────────────────────┬────────────────────────────────┘
                                  ▼
                  citation-grounding verifier  ◄── EvidenceIndex
                                  ▼
                CGS  ·  hallucination rate  ·  redacted output
```

# Reproducibility and evaluation

`ai-mutation` ships with an evaluation harness (`eval/`) that
(i) builds a stratified benchmark from CIViC *Accepted* evidence items,
(ii) runs three configurations of the same model — `no_rag` (zero-shot),
`rag_text` (full retrieval, text-only), `rag_mm` (retrieval +
multi-modal) — under deterministic settings (temperature = 0, fixed
seed), and (iii) reports the three headline metrics with 95 %
bootstrap confidence intervals: clinical-class agreement against the
CIViC ground truth, CGS, and hallucination rate. Per-stage prompt
hashes, model versions, seeds and token counts are persisted for every
run.

# Acknowledgements

We acknowledge the maintainers of CIViC, ClinVar, Open Targets,
MyGene.info / MyVariant.info, UniProt, AlphaFold, and Mutalyzer for the
public APIs that make this work possible.

# References
