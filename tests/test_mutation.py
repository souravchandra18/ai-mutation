"""Parser tests — pure, offline, deterministic."""
from __future__ import annotations

import pytest

from src.mutation import VariantClass, parse_mutation


CASES = [
    # (input, gene, variant_class, protein_change or None)
    ("BRAF V600E",                "BRAF", VariantClass.MISSENSE,   "V600E"),
    ("BRAF p.Val600Glu",          "BRAF", VariantClass.MISSENSE,   "V600E"),
    ("TP53 R213*",                "TP53", VariantClass.NONSENSE,   "R213*"),
    ("TP53 p.Arg213Ter",          "TP53", VariantClass.NONSENSE,   "R213*"),
    ("EGFR p.L858R",              "EGFR", VariantClass.MISSENSE,   "L858R"),
    ("EGFR L747_E749del",         "EGFR", VariantClass.INFRAME_DEL, None),
    ("EGFR D770_N771insSVD",      "EGFR", VariantClass.INFRAME_INS, None),
    ("EGFR Y772_A775dup",         "EGFR", VariantClass.INFRAME_DUP, None),
    ("BRCA1 K45Rfs*4",            "BRCA1", VariantClass.FRAMESHIFT, None),
    ("BRCA1 p.Lys45ArgfsTer4",    "BRCA1", VariantClass.FRAMESHIFT, None),
    ("BRCA1 c.5074+1G>A",         "BRCA1", VariantClass.SPLICE_SITE, None),
    ("MET exon14skip",            "MET",  VariantClass.EXON_SKIP,  None),
    ("MET ex14skip",              "MET",  VariantClass.EXON_SKIP,  None),
    ("HER2 amplification",        "HER2", VariantClass.CNV_AMP,    None),
    ("CDKN2A deletion",           "CDKN2A", VariantClass.CNV_DEL,  None),
    ("rs113488022",               None,   VariantClass.DBSNP,      None),
]


@pytest.mark.parametrize("text,gene,vc,pc", CASES)
def test_parse(text, gene, vc, pc):
    mq = parse_mutation(text)
    assert mq.variant_class == vc, f"{text}: got {mq.variant_class}"
    if gene is not None:
        assert mq.gene == gene
    if pc is not None:
        assert mq.protein_change == pc


def test_fusion():
    mq = parse_mutation("BCR::ABL1")
    assert mq.variant_class == VariantClass.FUSION
    assert mq.fusion_partners == ("BCR", "ABL1")


def test_fusion_dash():
    mq = parse_mutation("EML4-ALK")
    assert mq.variant_class == VariantClass.FUSION
    assert mq.fusion_partners == ("EML4", "ALK")


def test_compound():
    mq = parse_mutation("BRAF V600E;K601E")
    assert mq.variant_class == VariantClass.COMPOUND
    assert mq.compound_parts is not None
    assert len(mq.compound_parts) == 2
    assert mq.compound_parts[0].protein_change == "V600E"
    assert mq.compound_parts[1].protein_change == "K601E"
    # Gene inheritance
    assert mq.compound_parts[1].gene == "BRAF"


def test_hgvs_passthrough():
    mq = parse_mutation("NM_004333.6:c.1799T>A")
    assert mq.variant_class == VariantClass.HGVS
    assert mq.hgvs == "NM_004333.6:c.1799T>A"
