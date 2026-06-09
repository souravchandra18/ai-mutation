"""Parse user-supplied mutation strings into a canonical query.

Supported forms:
  Point mutations (1- or 3-letter AA codes)
    - "BRAF V600E"   "BRAF p.Val600Glu"
    - "TP53 R213*"   "TP53 p.Arg213Ter"
    - "EGFR p.L858R" "EGFR p.Leu858Arg"
  Indels / dup / frameshift
    - "EGFR L747_E749del"     "EGFR p.Leu747_Glu749del"
    - "EGFR D770_N771insSVD"  "EGFR p.Asp770_Asn771insSerValAsp"
    - "EGFR Y772_A775dup"     "BRCA1 K45Rfs*4"     "BRCA1 p.Lys45ArgfsTer4"
  Splice / coding HGVS (bare, with a gene)
    - "BRCA1 c.5074+1G>A"   "MET c.3028+1G>T"
  Exon-level
    - "MET exon14skip"   "MET ex14skip"
  Structural / copy-number
    - "BCR::ABL1"   "EML4-ALK"   "HER2 amplification"   "CDKN2A deletion"
  Identifiers (pass-through)
    - "NM_004333.6:c.1799T>A"   "rs113488022"
  Compound (semicolon-separated)
    - "BRAF V600E;K601E"   "TP53 R175H; R248Q"
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class VariantClass(str, Enum):
    """String-backed enum so existing `== "missense"` comparisons keep working."""

    MISSENSE = "missense"
    NONSENSE = "nonsense"
    SYNONYMOUS = "synonymous"
    INFRAME_DEL = "inframe_deletion"
    INFRAME_INS = "inframe_insertion"
    INFRAME_DUP = "inframe_duplication"
    FRAMESHIFT = "frameshift"
    SPLICE_SITE = "splice_site"
    EXON_SKIP = "exon_skip"
    FUSION = "fusion"
    CNV_AMP = "amplification"
    CNV_DEL = "cnv_deletion"
    HGVS = "hgvs"
    DBSNP = "dbsnp"
    GENE_ONLY = "gene_only"
    COMPOUND = "compound"
    UNKNOWN = "unknown"

    def __str__(self) -> str:  # JSON-friendly
        return self.value


# ---------------------------------------------------------------------------
# Amino-acid 3-letter → 1-letter normalisation
# ---------------------------------------------------------------------------
_AA3_TO_1: dict[str, str] = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
    "Sec": "U", "Pyl": "O",
    # stop codons
    "Ter": "*", "Stop": "*",
}

# Match a 3-letter code only when it stands as an AA token:
#   - not immediately preceded by an UPPERCASE letter (so we don't slice into
#     another Title-case AA code). Lowercase predecessors are allowed so that
#     codes following `fs`/`del`/`ins`/`dup` (e.g. `fsTer4`, `insSer`) still match.
#   - immediately followed by a digit, an uppercase letter (next AA),
#     end-of-string, or one of the HGVS keywords (fs/del/ins/dup/=).
_AA3_RE = re.compile(
    r"(?<![A-Z])"
    r"(Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|Leu|Lys|Met|Phe|Pro|Ser|"
    r"Thr|Trp|Tyr|Val|Sec|Pyl|Ter|Stop)"
    r"(?=\d|[A-Z]|fs|del|ins|dup|=|$)"
)


def _aa3_to_aa1(s: str) -> str:
    """Convert 3-letter amino-acid codes inside a protein change to 1-letter.

    Examples:
        Val600Glu        -> V600E
        Arg213Ter        -> R213*
        Lys45ArgfsTer4   -> K45Rfs*4
        Leu747_Glu749del -> L747_E749del
        Asp770_Asn771insSerValAsp -> D770_N771insSVD
    """
    return _AA3_RE.sub(lambda m: _AA3_TO_1[m.group(1)], s)


@dataclass
class MutationQuery:
    raw: str
    variant_class: str = VariantClass.UNKNOWN.value
    gene: str | None = None
    protein_change: str | None = None
    hgvs: str | None = None
    rsid: str | None = None
    fusion_partners: tuple[str, str] | None = None
    exon: int | None = None
    # Populated only for compound (semicolon-separated) inputs.
    compound_parts: list["MutationQuery"] | None = field(default=None)

    @property
    def label(self) -> str:
        if self.compound_parts:
            return "; ".join(p.label for p in self.compound_parts)
        if self.fusion_partners:
            return f"{self.fusion_partners[0]}::{self.fusion_partners[1]} fusion"
        if self.variant_class == VariantClass.EXON_SKIP and self.gene and self.exon:
            return f"{self.gene} exon {self.exon} skipping"
        if self.variant_class in (VariantClass.CNV_AMP, VariantClass.CNV_DEL) and self.gene:
            tag = "amplification" if self.variant_class == VariantClass.CNV_AMP else "deletion"
            return f"{self.gene} {tag}"
        if self.variant_class == VariantClass.SPLICE_SITE and self.gene and self.hgvs:
            return f"{self.gene} {self.hgvs}"
        if self.gene and self.protein_change:
            return f"{self.gene} {self.protein_change}"
        if self.gene:
            return self.gene
        return self.hgvs or self.rsid or self.raw


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------
_RS_RE = re.compile(r"^rs\d+$", re.IGNORECASE)
_HGVS_FULL_RE = re.compile(r"^[NX][MR]_\d+(?:\.\d+)?:[cgmnpr]\.", re.IGNORECASE)
# Bare HGVS (no transcript), used only when a gene precedes it: e.g. `c.5074+1G>A`.
_HGVS_BARE_RE = re.compile(r"^[cgmnpr]\.\S+$", re.IGNORECASE)
_SPLICE_RE = re.compile(r"c\.\d+[+\-]\d+", re.IGNORECASE)

_FUSION_RE = re.compile(r"^([A-Z][A-Z0-9]{1,15})(?:::|-)([A-Z][A-Z0-9]{1,15})$")
_SYMBOL_RE = re.compile(r"^[A-Z][A-Z0-9-]{0,15}$")

_PT_MISSENSE = re.compile(r"^([A-Z])(\d+)([A-Z])$")
_PT_NONSENSE = re.compile(r"^([A-Z])(\d+)([*X])$")
_PT_SYNONYM = re.compile(r"^([A-Z])(\d+)=$")
_PT_FS = re.compile(r"^([A-Z])(\d+)(?:[A-Z])?fs(?:\*\d+)?$")
_PT_DEL = re.compile(r"^([A-Z])(\d+)(?:_([A-Z])(\d+))?del[A-Z]*$")
_PT_INS = re.compile(r"^([A-Z])(\d+)_([A-Z])(\d+)ins[A-Z]+$")
_PT_DUP = re.compile(r"^([A-Z])(\d+)(?:_([A-Z])(\d+))?dup[A-Z]*$")

_EXON_RE = re.compile(r"^(?:ex|exon)\s*(\d+)\s*(?:skip|skipping)$", re.IGNORECASE)
_CNV_AMP_RE = re.compile(r"^(?:amp|amplification|amplified|gain)$", re.IGNORECASE)
_CNV_DEL_RE = re.compile(
    r"^(?:loss|deleted|deletion|homozygous\s+deletion|cnv-?del)$", re.IGNORECASE
)


def _classify_protein_change(change: str) -> tuple[str, str] | None:
    """Return (normalised_change, variant_class) or None if unrecognised.

    Accepts both 1-letter (V600E) and 3-letter (Val600Glu) inputs.
    """
    c = change.replace("p.", "").replace("P.", "").strip()
    # Normalise 3-letter → 1-letter codes BEFORE classification.
    c = _aa3_to_aa1(c)

    if _PT_NONSENSE.match(c):
        return c.replace("X", "*").upper(), VariantClass.NONSENSE.value
    if _PT_MISSENSE.match(c):
        return c.upper(), VariantClass.MISSENSE.value
    if _PT_SYNONYM.match(c):
        return c.upper(), VariantClass.SYNONYMOUS.value
    if _PT_FS.match(c):
        return c, VariantClass.FRAMESHIFT.value
    if _PT_INS.match(c):
        return c, VariantClass.INFRAME_INS.value
    if _PT_DUP.match(c):
        return c, VariantClass.INFRAME_DUP.value
    if _PT_DEL.match(c):
        return c, VariantClass.INFRAME_DEL.value
    return None


def _parse_single(text: str, default_gene: str | None = None) -> MutationQuery:
    """Parse a single (non-compound) mutation expression."""
    s = text.strip()
    if not s:
        raise ValueError("Empty mutation string")

    if _RS_RE.match(s):
        return MutationQuery(raw=s, rsid=s.lower(), variant_class=VariantClass.DBSNP.value)

    if _HGVS_FULL_RE.match(s):
        vclass = (
            VariantClass.SPLICE_SITE.value if _SPLICE_RE.search(s) else VariantClass.HGVS.value
        )
        return MutationQuery(raw=s, hgvs=s, variant_class=vclass)

    # Fusion (single whitespace-free token with `::` or `-` between two symbols)
    if " " not in s:
        fm = _FUSION_RE.match(s.upper())
        if fm:
            a, b = fm.group(1), fm.group(2)
            if _SYMBOL_RE.match(a) and _SYMBOL_RE.match(b):
                return MutationQuery(
                    raw=s, variant_class=VariantClass.FUSION.value, fusion_partners=(a, b)
                )

    parts = re.split(r"[\s:]+", s, maxsplit=1)
    gene: str | None
    rest: str | None

    if len(parts) == 2 and _SYMBOL_RE.match(parts[0].upper()):
        gene = parts[0].upper()
        rest = parts[1].strip()
    elif default_gene and len(parts) == 1:
        # Compound part lacking its own gene; inherit from the first part.
        gene = default_gene
        rest = parts[0].strip()
    else:
        gene = None
        rest = None

    if gene and rest:
        em = _EXON_RE.match(rest)
        if em:
            return MutationQuery(
                raw=s, gene=gene,
                variant_class=VariantClass.EXON_SKIP.value, exon=int(em.group(1))
            )
        if _CNV_AMP_RE.match(rest):
            return MutationQuery(raw=s, gene=gene, variant_class=VariantClass.CNV_AMP.value)
        if _CNV_DEL_RE.match(rest):
            return MutationQuery(raw=s, gene=gene, variant_class=VariantClass.CNV_DEL.value)

        # Try protein-change classification BEFORE bare HGVS so `p.Val600Glu`
        # (which would syntactically match `^p\..*`) is correctly normalised
        # to `V600E` / MISSENSE instead of being passed through as raw HGVS.
        classified = _classify_protein_change(rest)
        if classified:
            change, vclass = classified
            return MutationQuery(
                raw=s, gene=gene, protein_change=change, variant_class=vclass
            )

        # Bare HGVS (c./g./n./m./r./p.) after the gene, e.g. "BRCA1 c.5074+1G>A".
        if _HGVS_BARE_RE.match(rest):
            vclass = (
                VariantClass.SPLICE_SITE.value
                if _SPLICE_RE.search(rest)
                else VariantClass.HGVS.value
            )
            return MutationQuery(raw=s, gene=gene, hgvs=rest, variant_class=vclass)

    upper = s.upper()
    if _SYMBOL_RE.match(upper):
        return MutationQuery(raw=s, gene=upper, variant_class=VariantClass.GENE_ONLY.value)

    return MutationQuery(raw=s, variant_class=VariantClass.UNKNOWN.value)


def parse_mutation(text: str) -> MutationQuery:
    """Parse a mutation string. Supports compound (semicolon-separated) forms.

    For compound input like `"BRAF V600E;K601E"` the returned MutationQuery has
    `variant_class = "compound"`, `compound_parts` filled with each parsed
    sub-query, and `gene` set when every part shares one gene.
    """
    s = (text or "").strip()
    if not s:
        raise ValueError("Empty mutation string")

    # Compound: split only on `;` (commas appear inside HGVS / protein changes too often).
    if ";" in s:
        chunks = [c.strip() for c in s.split(";") if c.strip()]
        if len(chunks) >= 2:
            first = _parse_single(chunks[0])
            parts: list[MutationQuery] = [first]
            inherit_gene = first.gene
            for chunk in chunks[1:]:
                parts.append(_parse_single(chunk, default_gene=inherit_gene))
            shared_gene = (
                parts[0].gene
                if parts[0].gene and all(p.gene == parts[0].gene for p in parts)
                else None
            )
            return MutationQuery(
                raw=s,
                variant_class=VariantClass.COMPOUND.value,
                gene=shared_gene,
                compound_parts=parts,
            )

    return _parse_single(s)
