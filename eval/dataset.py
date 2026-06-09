"""Benchmark dataset builder.

Pulls a stratified sample of CIViC *Accepted* evidence items and writes
them to `eval/data/civic_benchmark.jsonl`. Each record is a
ground-truth label that can be compared against the LLM's stage-3
output.

Run as:
    python -m eval.dataset --n 50 --out eval/data/civic_benchmark.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import httpx

CIVIC_GQL = "https://civicdb.org/api/graphql"

# Stratification targets — taken from clinical CDS literature; map CIViC
# significance values to four broad classes we score against.
SIG_CLASSES: dict[str, str] = {
    "SENSITIVITYRESPONSE": "sensitivity",
    "RESISTANCE": "resistance",
    "PATHOGENIC": "pathogenic",
    "LIKELY_PATHOGENIC": "pathogenic",
    "BENIGN": "benign",
    "LIKELY_BENIGN": "benign",
}

QUERY = """
query Browse($after: String) {
  evidenceItems(status: ACCEPTED, first: 50, after: $after) {
    pageInfo { hasNextPage endCursor }
    nodes {
      id
      evidenceType
      evidenceLevel
      significance
      description
      source { citationId sourceType }
      molecularProfile {
        name
        variants { name feature { name } }
      }
      disease { name doid }
      therapies { name }
    }
  }
}
"""


def _fetch_page(after: str | None) -> dict[str, Any]:
    r = httpx.post(
        CIVIC_GQL,
        json={"query": QUERY, "variables": {"after": after}},
        timeout=30.0,
    )
    r.raise_for_status()
    return r.json()["data"]["evidenceItems"]


def _to_record(ev: dict[str, Any]) -> dict[str, Any] | None:
    sig = (ev.get("significance") or "").upper()
    cls = SIG_CLASSES.get(sig)
    if not cls:
        return None
    mp = ev.get("molecularProfile") or {}
    variants = mp.get("variants") or []
    if not variants:
        return None
    v = variants[0]
    gene = ((v.get("feature") or {}).get("name") or "").strip()
    var_name = (v.get("name") or "").strip()
    if not gene or not var_name:
        return None
    return {
        "civic_evidence_id": f"EID{ev['id']}",
        "gene": gene,
        "variant": var_name,
        "label": f"{gene} {var_name}",
        "ground_truth_class": cls,
        "ground_truth_significance": sig,
        "evidence_type": ev.get("evidenceType"),
        "evidence_level": ev.get("evidenceLevel"),
        "disease": (ev.get("disease") or {}).get("name"),
        "therapies": [(t or {}).get("name") for t in (ev.get("therapies") or [])],
        "pubmed_id": ((ev.get("source") or {}).get("citationId")
                      if (ev.get("source") or {}).get("sourceType") == "PUBMED"
                      else None),
        "description_excerpt": (ev.get("description") or "")[:400],
    }


def build(n: int, out: Path, max_pages: int = 60, seed: int = 7) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)

    by_class: dict[str, list[dict[str, Any]]] = {c: [] for c in set(SIG_CLASSES.values())}
    after: str | None = None
    pages = 0
    while pages < max_pages:
        page = _fetch_page(after)
        for ev in page["nodes"]:
            rec = _to_record(ev)
            if rec:
                by_class[rec["ground_truth_class"]].append(rec)
        if not page["pageInfo"]["hasNextPage"]:
            break
        after = page["pageInfo"]["endCursor"]
        pages += 1
        # Stop early if we already have enough headroom in each class.
        if all(len(v) >= max(4, n // 2) for v in by_class.values()):
            break

    # Stratified sample.
    per_class = max(1, n // len(by_class))
    chosen: list[dict[str, Any]] = []
    for cls, pool in by_class.items():
        rng.shuffle(pool)
        chosen.extend(pool[:per_class])
    rng.shuffle(chosen)
    chosen = chosen[:n]

    with out.open("w") as fh:
        for rec in chosen:
            fh.write(json.dumps(rec) + "\n")

    return len(chosen)


def main() -> None:
    p = argparse.ArgumentParser(description="Build CIViC benchmark dataset")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--out", type=Path,
                   default=Path("eval/data/civic_benchmark.jsonl"))
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    written = build(args.n, args.out, seed=args.seed)
    print(f"Wrote {written} records to {args.out}", flush=True)
    print(f"  class distribution:")
    counts: dict[str, int] = {}
    for line in args.out.read_text().splitlines():
        c = json.loads(line)["ground_truth_class"]
        counts[c] = counts.get(c, 0) + 1
    for k, v in sorted(counts.items()):
        print(f"    {k:<15s} {v}")


if __name__ == "__main__":
    main()
