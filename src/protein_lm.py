"""ESM-2 protein language model — second model + sequence modality.

For a missense variant we compute a zero-shot variant-effect score by
masking the wild-type residue and reading off the log-likelihood of the
WT vs mutant amino acid:

    delta_pll = log P(mutant | context) - log P(WT | context)

Strongly negative values indicate the model considers the substitution
unlikely under the natural sequence distribution and is therefore a
candidate for loss / disrupting effect; strongly positive values are
unusual and may suggest gain-of-function in the right structural context.

This is a real second model running on the GPU in-process (HuggingFace
transformers on PyTorch-ROCm). Heavy imports are lazy so callers without
torch/transformers fall through to `{"found": False, "reason": ...}`.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from .http import get_json
from .mutation import MutationQuery, VariantClass

UNIPROT_FASTA = "https://rest.uniprot.org/uniprotkb/{acc}.fasta"
MYGENE = "https://mygene.info/v3"

DEFAULT_MODEL = os.getenv("ESM2_MODEL", "facebook/esm2_t33_650M_UR50D")
ENABLED = os.getenv("ESM2_ENABLED", "1") != "0"
MAX_LEN = int(os.getenv("ESM2_MAX_LEN", "1022"))  # ESM-2 hard limit


# ---------------------------------------------------------------------------
# Lazy model loading — keeps import-time cost zero and makes the dependency
# optional. The first call pays a few seconds; subsequent calls reuse.
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_model() -> tuple[Any, Any, Any] | None:
    try:
        import torch  # type: ignore
        from transformers import AutoModelForMaskedLM, AutoTokenizer  # type: ignore
    except Exception:  # noqa: BLE001
        return None
    try:
        tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
        model = AutoModelForMaskedLM.from_pretrained(DEFAULT_MODEL)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device).eval()
        return model, tok, device
    except Exception:  # noqa: BLE001
        return None


def _uniprot_acc(symbol: str) -> str | None:
    try:
        res = get_json(
            f"{MYGENE}/query",
            params={"q": f"symbol:{symbol}", "species": "human",
                    "size": 1, "fields": "uniprot"},
        )
        hits = res.get("hits") or []
        if not hits:
            return None
        up = hits[0].get("uniprot") or {}
        sp = up.get("Swiss-Prot")
        if isinstance(sp, list):
            return sp[0] if sp else None
        return sp if isinstance(sp, str) else None
    except Exception:  # noqa: BLE001
        return None


def _fetch_fasta_sequence(acc: str) -> str | None:
    try:
        import httpx
        r = httpx.get(UNIPROT_FASTA.format(acc=acc), timeout=15.0)
        r.raise_for_status()
        lines = [ln for ln in r.text.splitlines() if not ln.startswith(">")]
        seq = "".join(lines).strip()
        return seq or None
    except Exception:  # noqa: BLE001
        return None


def _classify(delta: float) -> str:
    """Coarse bucket from delta_pll."""
    if delta <= -4.0:
        return "very likely disrupting"
    if delta <= -2.0:
        return "likely disrupting"
    if delta < 1.0:
        return "uncertain / mild"
    if delta < 3.0:
        return "tolerated / possibly activating"
    return "unusual — context-dependent (consider gain-of-function)"


def _missense_residues(pc: str | None) -> tuple[str, int, str] | None:
    """Extract (WT, position, MUT) from a 1-letter missense protein change."""
    if not pc or len(pc) < 3:
        return None
    wt = pc[0]
    mut = pc[-1]
    middle = pc[1:-1]
    if not (wt.isalpha() and mut.isalpha() and middle.isdigit()):
        return None
    return wt.upper(), int(middle), mut.upper()


def _score_missense(model: Any, tok: Any, device: str,
                    seq: str, residue_idx: int, wt: str, mut: str) -> dict[str, Any]:
    """Compute log P(mut|ctx) - log P(wt|ctx) at position residue_idx (1-based)."""
    import torch  # type: ignore

    if residue_idx < 1 or residue_idx > len(seq):
        return {"found": False, "reason": "residue index out of range"}
    if seq[residue_idx - 1].upper() != wt:
        return {
            "found": False,
            "reason": f"WT mismatch: sequence has {seq[residue_idx-1]} at "
                      f"position {residue_idx}, query expects {wt}",
        }

    # ESM-2 input: a single sequence; mask the residue, ask the LM for logits.
    # Slice a window around the mutated residue if the protein is too long.
    half = MAX_LEN // 2
    start = max(0, residue_idx - 1 - half)
    end = min(len(seq), start + MAX_LEN)
    start = max(0, end - MAX_LEN)
    sub = seq[start:end]
    mask_pos_in_sub = residue_idx - 1 - start  # 0-based within sub

    masked = sub[:mask_pos_in_sub] + tok.mask_token + sub[mask_pos_in_sub + 1:]
    enc = tok(masked, return_tensors="pt").to(device)
    # Find the index of the mask token in the tokenised input.
    mask_token_id = tok.mask_token_id
    mask_index = (enc["input_ids"][0] == mask_token_id).nonzero(as_tuple=True)[0]
    if mask_index.numel() == 0:
        return {"found": False, "reason": "tokeniser dropped the mask"}
    mask_index = int(mask_index[0])

    with torch.no_grad():
        logits = model(**enc).logits[0, mask_index]
        log_probs = torch.log_softmax(logits, dim=-1)

    wt_id = tok.convert_tokens_to_ids(wt)
    mut_id = tok.convert_tokens_to_ids(mut)
    if wt_id is None or mut_id is None or wt_id == tok.unk_token_id \
            or mut_id == tok.unk_token_id:
        return {"found": False, "reason": "AA token id unavailable"}

    wt_lp = float(log_probs[wt_id])
    mut_lp = float(log_probs[mut_id])
    delta = mut_lp - wt_lp

    # Top-5 most likely AAs at the masked position — useful sanity signal.
    top_vals, top_idx = log_probs.topk(5)
    top_aas: list[dict[str, Any]] = []
    for v, i in zip(top_vals.tolist(), top_idx.tolist()):
        token = tok.convert_ids_to_tokens(int(i))
        if token and len(token) == 1 and token.isalpha():
            top_aas.append({"aa": token, "log_prob": round(float(v), 4)})

    return {
        "found": True,
        "model": DEFAULT_MODEL,
        "wt": wt,
        "mut": mut,
        "position": residue_idx,
        "wt_log_prob": round(wt_lp, 4),
        "mut_log_prob": round(mut_lp, 4),
        "delta_pll": round(delta, 4),
        "classification": _classify(delta),
        "top5_predicted_aa": top_aas,
        "device": device,
        "sequence_length": len(seq),
    }


def score_variant(mq: MutationQuery) -> dict[str, Any]:
    """Run ESM-2 on a missense variant. Returns a JSON-serialisable dict."""
    if not ENABLED:
        return {"found": False, "reason": "ESM2_ENABLED=0"}
    if mq.variant_class != VariantClass.MISSENSE.value or not mq.gene:
        return {"found": False, "reason": f"not applicable for {mq.variant_class}"}
    parsed = _missense_residues(mq.protein_change)
    if not parsed:
        return {"found": False, "reason": "could not parse protein change"}

    acc = _uniprot_acc(mq.gene)
    if not acc:
        return {"found": False, "reason": "no UniProt accession"}
    seq = _fetch_fasta_sequence(acc)
    if not seq:
        return {"found": False, "reason": "no UniProt FASTA"}

    loaded = _load_model()
    if loaded is None:
        return {
            "found": False,
            "reason": (
                "transformers/torch unavailable — install with "
                "`pip install transformers torch` (ROCm wheel on AMD)."
            ),
        }
    model, tok, device = loaded

    wt, pos, mut = parsed
    try:
        return _score_missense(model, tok, device, seq, pos, wt, mut)
    except Exception as e:  # noqa: BLE001
        return {"found": False, "reason": f"ESM-2 inference failed: {e}"}
