"""Score and aggregate benchmark results into a publishable report.

Reads all `eval/results/<mode>.jsonl` files written by `run_benchmark`
and emits a Markdown table with per-mode means and 95 % bootstrap
confidence intervals on the headline metrics.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from statistics import mean


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _bootstrap_ci(xs: list[float], n_iter: int = 1000, seed: int = 7) -> tuple[float, float]:
    if not xs:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = []
    for _ in range(n_iter):
        sample = [xs[rng.randrange(len(xs))] for _ in range(len(xs))]
        means.append(sum(sample) / len(sample))
    means.sort()
    lo = means[int(0.025 * n_iter)]
    hi = means[int(0.975 * n_iter)]
    return (lo, hi)


def _summarise(records: list[dict]) -> dict:
    if not records:
        return {}
    valid = [r for r in records if "error" not in r]
    n = len(valid)
    agreement = [r["agreement"] for r in valid]
    cgs = [r.get("grounding", {}).get("citation_grounding_score", 0.0) for r in valid]
    hall = [r.get("grounding", {}).get("hallucination_rate", 0.0) for r in valid]
    total_cites = [r.get("grounding", {}).get("total_citations", 0) for r in valid]
    elapsed = []
    for r in valid:
        stages = (r.get("run") or {}).get("stages", {})
        elapsed.append(sum(s.get("elapsed_s", 0.0) or 0.0 for s in stages.values()))
    tokens = []
    for r in valid:
        stages = (r.get("run") or {}).get("stages", {})
        tokens.append(sum((s.get("total_tokens") or 0) for s in stages.values()))

    ci_agree = _bootstrap_ci([float(a) for a in agreement])
    ci_cgs = _bootstrap_ci(cgs)
    ci_hall = _bootstrap_ci(hall)

    return {
        "n": n,
        "errors": len(records) - n,
        "agreement_mean": mean(agreement) if agreement else 0.0,
        "agreement_ci": ci_agree,
        "cgs_mean": mean(cgs) if cgs else 0.0,
        "cgs_ci": ci_cgs,
        "hall_mean": mean(hall) if hall else 0.0,
        "hall_ci": ci_hall,
        "citations_mean": mean(total_cites) if total_cites else 0.0,
        "elapsed_mean": mean(elapsed) if elapsed else 0.0,
        "tokens_mean": mean(tokens) if tokens else 0.0,
    }


def _fmt(v: float, d: int = 3) -> str:
    if math.isnan(v):
        return "—"
    return f"{v:.{d}f}"


def _fmt_ci(ci: tuple[float, float], d: int = 3) -> str:
    return f"[{_fmt(ci[0], d)}, {_fmt(ci[1], d)}]"


def build_report(results_dir: Path) -> str:
    modes = ["no_rag", "rag_text", "rag_mm"]
    rows = {m: _summarise(_load(results_dir / f"{m}.jsonl")) for m in modes}

    lines: list[str] = []
    lines.append("# ai-mutation benchmark — aggregate report\n")
    lines.append(f"_Results directory: `{results_dir}`_\n")
    lines.append("\n## Headline metrics\n")
    lines.append(
        "| Mode | n | Agreement | 95 % CI | CGS | 95 % CI | Hallucination | 95 % CI | Avg cites | Avg tokens | Avg latency (s) |"
    )
    lines.append(
        "|---|---:|---:|---|---:|---|---:|---|---:|---:|---:|"
    )
    for m in modes:
        s = rows[m]
        if not s:
            lines.append(f"| `{m}` | 0 | — | — | — | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| `{m}` | {s['n']} | "
            f"{_fmt(s['agreement_mean'])} | {_fmt_ci(s['agreement_ci'])} | "
            f"{_fmt(s['cgs_mean'])} | {_fmt_ci(s['cgs_ci'])} | "
            f"{_fmt(s['hall_mean'])} | {_fmt_ci(s['hall_ci'])} | "
            f"{_fmt(s['citations_mean'], 1)} | "
            f"{_fmt(s['tokens_mean'], 0)} | "
            f"{_fmt(s['elapsed_mean'], 2)} |"
        )

    lines.append("\n## Legend\n")
    lines.append(
        "- **Agreement** — fraction of variants where the LLM's predicted "
        "clinical class matches the curated CIViC class.\n"
        "- **CGS** (Citation-Grounding Score) — fraction of LLM-emitted "
        "citations that resolve to a source / identifier present in the "
        "retrieved evidence.\n"
        "- **Hallucination rate** — fraction of citations that are "
        "fabricated identifiers (PMIDs / EIDs the retrieval layer never "
        "saw).\n"
        "- 95 % CIs are bootstrap (1 000 resamples, seed=7).\n"
    )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("results_dir", type=Path, nargs="?",
                   default=Path("eval/results"))
    p.add_argument("--out", type=Path, default=Path("eval/results/REPORT.md"))
    args = p.parse_args()

    report = build_report(args.results_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report)
    print(report)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
