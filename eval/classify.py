"""Outcome-classification heuristic.

Given an LLM stage-3 string, predict which of the four CIViC clinical
classes ({sensitivity, resistance, pathogenic, benign}) it best
expresses. This is intentionally simple and rule-based so the
evaluation does not depend on a *second* LLM judge (which would
re-introduce the very hallucination problem we are measuring).
"""
from __future__ import annotations

import re
from typing import Iterable

_KEYWORDS: dict[str, tuple[str, ...]] = {
    "sensitivity": (
        "sensitive", "sensitivity", "response", "responsive", "responder",
        "benefit", "efficacious", "approved", "fda-approved",
    ),
    "resistance": (
        "resistance", "resistant", "refractory", "lack of response",
        "loss of efficacy", "fails to respond",
    ),
    "pathogenic": (
        "pathogenic", "likely pathogenic", "oncogenic", "driver",
        "loss-of-function", "gain-of-function", "deleterious",
        "damaging", "tumour suppressor loss", "tumor suppressor loss",
    ),
    "benign": (
        "benign", "likely benign", "polymorphism", "neutral", "tolerated",
        "no functional impact", "not deleterious",
    ),
}


def _score(text: str, words: Iterable[str]) -> int:
    t = text.lower()
    return sum(len(re.findall(rf"\b{re.escape(w)}\b", t)) for w in words)


def classify(text: str) -> tuple[str, dict[str, int]]:
    """Return (predicted_class, raw_keyword_counts)."""
    scores = {cls: _score(text, kws) for cls, kws in _KEYWORDS.items()}
    # Tie-breaking: pathogenic > sensitivity > resistance > benign (mirrors
    # the prior in oncology variant curation).
    order = ("pathogenic", "sensitivity", "resistance", "benign")
    best = max(order, key=lambda c: (scores[c], -order.index(c)))
    if scores[best] == 0:
        return "uncertain", scores
    return best, scores


def agreement(pred: str, truth: str) -> int:
    """1 if the predicted class equals the curated class, else 0.
    `uncertain` predictions count as disagreement."""
    return int(pred == truth)
