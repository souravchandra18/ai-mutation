"""Benchmark runner.

For each variant in the benchmark JSONL, run one or more of the three
configurations (no_rag / rag_text / rag_mm), persist the result as a
JSONL record, and tee a brief progress line to stdout.

Run as:
    python -m eval.run_benchmark \
        --variants eval/data/civic_benchmark.jsonl \
        --modes no_rag rag_text rag_mm \
        --out eval/results/

Results live under `eval/results/<mode>.jsonl` (append-only, resumable).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from src.evidence import gather
from src.llm_config import DEFAULT_VISION_MODEL, get_llm_settings
from src.reasoning import reason
from eval.classify import agreement, classify

MODES = ("no_rag", "rag_text", "rag_mm")


def _already_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            done.add(json.loads(line)["label"])
        except Exception:  # noqa: BLE001
            continue
    return done


def run_one(record: dict, mode: str, model: str | None, seed: int) -> dict:
    label = record["label"]
    t0 = time.perf_counter()
    mq, ev = gather(label)
    retrieval_s = time.perf_counter() - t0

    chosen_model = model
    no_rag = mode == "no_rag"
    if mode == "rag_mm" and not chosen_model:
        settings = get_llm_settings()
        chosen_model = settings.vision_model or DEFAULT_VISION_MODEL

    res = reason(
        mq, ev,
        model=chosen_model,
        deterministic=True,
        verify=True,
        redact=False,
        no_rag=no_rag,
        seed=seed,
    )

    pred_class, kw = classify(res.therapy + "\n" + res.mutation_summary)
    truth = record["ground_truth_class"]

    return {
        "label": label,
        "mode": mode,
        "model": chosen_model or get_llm_settings().model,
        "seed": seed,
        "retrieval_s": round(retrieval_s, 3),
        "predicted_class": pred_class,
        "ground_truth_class": truth,
        "agreement": agreement(pred_class, truth),
        "keyword_scores": kw,
        "grounding": res.grounding.get("aggregate", {}),
        "grounding_per_stage": res.grounding.get("per_stage", {}),
        "run": res.run,
        "outputs": {
            "mutation_summary": res.mutation_summary,
            "mechanism": res.mechanism,
            "therapy": res.therapy,
        },
        "ground_truth_record": record,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variants", type=Path,
                   default=Path("eval/data/civic_benchmark.jsonl"))
    p.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    p.add_argument("--out", type=Path, default=Path("eval/results/"))
    p.add_argument("--model", type=str, default=None,
                   help="Override LLM model (defaults to AI_MODEL / AMD_MODEL env)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N records (for smoke runs)")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    if not args.variants.exists():
        print(f"Benchmark file not found: {args.variants}", file=sys.stderr)
        print("Run `python -m eval.dataset` first.", file=sys.stderr)
        sys.exit(2)

    records = [json.loads(line) for line in args.variants.read_text().splitlines()
               if line.strip()]
    if args.limit:
        records = records[:args.limit]

    for mode in args.modes:
        out_path = args.out / f"{mode}.jsonl"
        done = _already_done(out_path)
        print(f"\n=== mode={mode}  resume={len(done)}/{len(records)} ===", flush=True)
        with out_path.open("a") as fh:
            for i, rec in enumerate(records, 1):
                if rec["label"] in done:
                    continue
                try:
                    out = run_one(rec, mode=mode, model=args.model, seed=args.seed)
                except Exception as e:  # noqa: BLE001
                    out = {
                        "label": rec["label"], "mode": mode,
                        "error": str(e), "ground_truth_record": rec,
                    }
                fh.write(json.dumps(out, default=str) + "\n")
                fh.flush()
                gs = out.get("grounding", {}) or {}
                print(
                    f"[{mode}] {i:>3}/{len(records)}  {rec['label']:<32s} "
                    f"pred={out.get('predicted_class','-'):<11s} "
                    f"truth={rec['ground_truth_class']:<11s} "
                    f"cgs={gs.get('citation_grounding_score','-')} "
                    f"hall={gs.get('hallucination_rate','-')}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
