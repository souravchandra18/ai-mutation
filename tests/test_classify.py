"""Tests for the rule-based outcome classifier used in the benchmark."""
from __future__ import annotations

from eval.classify import agreement, classify


def test_sensitivity():
    text = "BRAF V600E confers sensitivity to vemurafenib; patients respond."
    pred, _ = classify(text)
    assert pred == "sensitivity"
    assert agreement(pred, "sensitivity") == 1


def test_resistance():
    text = "EGFR T790M mediates resistance to gefitinib and is refractory."
    pred, _ = classify(text)
    assert pred == "resistance"


def test_pathogenic():
    text = "Loss-of-function TP53 variant; deleterious, oncogenic driver."
    pred, _ = classify(text)
    assert pred == "pathogenic"


def test_benign():
    text = "This variant is benign and tolerated; no functional impact."
    pred, _ = classify(text)
    assert pred == "benign"


def test_uncertain_when_no_signal():
    text = "Lorem ipsum dolor sit amet."
    pred, _ = classify(text)
    assert pred == "uncertain"
    assert agreement(pred, "sensitivity") == 0
