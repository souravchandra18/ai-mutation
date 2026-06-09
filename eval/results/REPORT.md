# ai-mutation benchmark — aggregate report

_Results directory: `eval\results`_


## Headline metrics

| Mode | n | Agreement | 95 % CI | CGS | 95 % CI | Hallucination | 95 % CI | Avg cites | Avg tokens | Avg latency (s) |
|---|---:|---:|---|---:|---|---:|---|---:|---:|---:|
| `no_rag` | 7 | 0.286 | [0.000, 0.714] | 0.435 | [0.227, 0.643] | 0.104 | [0.029, 0.165] | 22.0 | 3439 | 44.22 |
| `rag_text` | 0 | 0.000 | [0.000, 0.000] | 0.000 | [0.000, 0.000] | 0.000 | [0.000, 0.000] | 0.0 | 0 | 0.00 |
| `rag_mm` | 250 | 0.324 | [0.268, 0.380] | 0.433 | [0.373, 0.494] | 0.032 | [0.020, 0.046] | 7.9 | 9566 | 91.55 |

## Legend

- **Agreement** — fraction of variants where the LLM's predicted clinical class matches the curated CIViC class.
- **CGS** (Citation-Grounding Score) — fraction of LLM-emitted citations that resolve to a source / identifier present in the retrieved evidence.
- **Hallucination rate** — fraction of citations that are fabricated identifiers (PMIDs / EIDs the retrieval layer never saw).
- 95 % CIs are bootstrap (1 000 resamples, seed=7).
