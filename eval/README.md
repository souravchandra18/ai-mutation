# Evaluation harness

This directory contains the **research-mode benchmark** for the
`ai-mutation` pipeline. It measures three things:

1. **Citation grounding** — the fraction of LLM-emitted citations that
   point at an identifier or source actually present in the retrieved
   evidence (the inverse is the hallucination rate).
2. **Retrieval lift** — agreement with curated CIViC clinical
   significance, compared between three configurations of the same model:
   * `no_rag` — zero-shot, no evidence
   * `rag_text` — full retrieval, text-only prompt
   * `rag_mm` — full retrieval + multimodal (domain-map image)
3. **Cost / latency** — total tokens and wall-clock per variant.

## Quick start

```bash
# 1. Build the benchmark set (cached to eval/data/civic_benchmark.jsonl)
python -m eval.dataset --n 50

# 2. Run the three baselines (deterministic, seed=7)
python -m eval.run_benchmark --variants eval/data/civic_benchmark.jsonl \
    --modes no_rag rag_text rag_mm \
    --out eval/results/

# 3. Aggregate scores into a markdown report
python -m eval.score eval/results/ --out eval/results/REPORT.md
```

All runs write append-only JSONL so you can resume after a crash. Every
record includes the prompt hash, model, seed, and per-stage token
counts — sufficient for full reproducibility.

## Methodology summary

* **Ground truth** — CIViC *Accepted* evidence items for variants whose
  `clinical_significance` is one of {Sensitivity/Response, Resistance,
  Pathogenic, Likely Pathogenic, Benign, Likely Benign}. We sample a
  stratified subset of `--n` variants across all clinical-significance
  classes.
* **Citation grounding** — implemented in `src/verification.py`. A
  citation is *grounded* iff (a) the source tag has a non-empty
  retrieved-evidence block, **and** (b) any explicit identifier resolves
  inside that block.
* **Reproducibility** — `deterministic=True` sets `temperature=0` and
  forwards a fixed `seed` to the NIM endpoint. Prompt hashes are logged
  per stage.
* **Sample size** — for the published study, n ≥ 200 with stratified
  sampling; for development iteration, n = 20–50 is sufficient.

## Caveats

* CIViC evidence drift: re-run `dataset.py` to refresh the benchmark
  whenever you compare across time.
* LLM endpoints are not perfectly deterministic even with `seed`. We
  report mean ± std over 3 repeats per variant in the published numbers.
* No patient-derived data is ever sent to the LLM; only public,
  pre-curated knowledge-base records.
