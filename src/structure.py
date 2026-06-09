"""Structural context — the multi-modal layer.

For a single-gene variant we resolve:
  * The canonical UniProt accession (via MyGene.info)
  * Domain / region / active-site features (via UniProt REST)
  * The AlphaFold predicted structure (PDB URL + viewer URL)
  * A rendered PNG of the UniProt domain map with the mutated residue marked

The PNG is base64-encoded and surfaced under
`Evidence.structure["domain_plot_png_b64"]`. Vision-capable
OpenAI-compatible models (for example `Qwen/Qwen2.5-VL-7B-Instruct`)
receive it as an
`image_url` content part during the mechanism stage; text-only models
still get the structural context as JSON.
"""
from __future__ import annotations

import base64
import io
import os
import re
from typing import Any

from .http import get_json
from .mutation import MutationQuery, VariantClass

MYGENE = "https://mygene.info/v3"
UNIPROT = "https://rest.uniprot.org/uniprotkb"
ALPHAFOLD_API = "https://alphafold.ebi.ac.uk/api/prediction"
ALPHAFOLD_PDB_FALLBACK = "https://alphafold.ebi.ac.uk/files/AF-{acc}-F1-model_v6.pdb"
ALPHAFOLD_VIEWER = "https://alphafold.ebi.ac.uk/entry/{acc}"

INCLUDE_DOMAIN_PLOT = os.getenv("STRUCTURE_DOMAIN_PLOT", "1") != "0"
INCLUDE_3D_PLOT = os.getenv("STRUCTURE_3D_PLOT", "1") != "0"
MAX_FEATURES_IN_PROMPT = int(os.getenv("STRUCTURE_MAX_FEATURES", "25"))

# UniProt feature types we keep for the LLM / plot.
_DOMAIN_LIKE = {"Domain", "Region", "Zinc finger", "Repeat",
                "Transmembrane", "Coiled coil", "DNA binding"}
_SITE_LIKE = {"Active site", "Binding site", "Site"}
_OTHER_KEEP = {"Motif", "Signal peptide"}
_KEEP_TYPES = _DOMAIN_LIKE | _SITE_LIKE | _OTHER_KEEP


def _residue_from_query(mq: MutationQuery) -> int | None:
    pc = mq.protein_change
    if not pc:
        return None
    m = re.search(r"(\d+)", pc)
    return int(m.group(1)) if m else None


def _uniprot_accession(symbol: str) -> str | None:
    try:
        res = get_json(
            f"{MYGENE}/query",
            params={
                "q": f"symbol:{symbol}",
                "species": "human",
                "size": 1,
                "fields": "uniprot",
            },
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


def _uniprot_record(acc: str) -> dict[str, Any] | None:
    try:
        return get_json(f"{UNIPROT}/{acc}.json")
    except Exception:  # noqa: BLE001
        return None


def _alphafold_pdb_url(acc: str) -> str:
    """Resolve the current AlphaFold PDB URL via their prediction API.

    AlphaFold rolls the model version periodically (v2 → ... → v6 → ...).
    Hardcoding the suffix gives a 404 the moment they re-release. Falls
    back to the latest known suffix if the API is unreachable.
    """
    try:
        data = get_json(f"{ALPHAFOLD_API}/{acc}")
        if isinstance(data, list) and data:
            url = data[0].get("pdbUrl")
            if isinstance(url, str) and url:
                return url
    except Exception:  # noqa: BLE001
        pass
    return ALPHAFOLD_PDB_FALLBACK.format(acc=acc)


def _trim_features(record: dict[str, Any]) -> tuple[int | None, list[dict[str, Any]]]:
    length = (record.get("sequence") or {}).get("length")
    out: list[dict[str, Any]] = []
    for feat in record.get("features") or []:
        ftype = feat.get("type")
        if ftype not in _KEEP_TYPES:
            continue
        loc = feat.get("location") or {}
        s = (loc.get("start") or {}).get("value")
        e = (loc.get("end") or {}).get("value")
        if s is None or e is None:
            continue
        out.append({
            "type": ftype,
            "start": s,
            "end": e,
            "description": (feat.get("description") or "").strip(),
        })
    return length, out


def _domain_at(residue: int, features: list[dict[str, Any]]) -> dict[str, Any] | None:
    for f in features:
        if f["type"] in _DOMAIN_LIKE and f["start"] <= residue <= f["end"]:
            return f
    return None


def _nearby_sites(residue: int, features: list[dict[str, Any]],
                  window: int = 10) -> list[dict[str, Any]]:
    return [
        f for f in features
        if f["type"] in _SITE_LIKE and abs(f["start"] - residue) <= window
    ]


def _render_plot(symbol: str, length: int,
                 features: list[dict[str, Any]],
                 residue: int | None) -> str | None:
    """Render a domain-track + lollipop PNG, return base64 string."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure
        from matplotlib.patches import Rectangle
    except Exception:  # noqa: BLE001
        return None

    try:
        fig = Figure(figsize=(10, 2.4), dpi=120)
        ax = fig.add_subplot(111)
        ax.set_xlim(0, max(length, 1))
        ax.set_ylim(-1.5, 1.8)
        ax.axhline(0, color="#bbb", lw=2, zorder=1)

        palette = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
                   "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD"]
        domain_like = [f for f in features if f["type"] in _DOMAIN_LIKE]
        for i, f in enumerate(domain_like):
            width = max(f["end"] - f["start"], 1)
            ax.add_patch(Rectangle(
                (f["start"], -0.4), width, 0.8,
                facecolor=palette[i % len(palette)], edgecolor="black",
                linewidth=0.5, alpha=0.85, zorder=2,
            ))
            mid = (f["start"] + f["end"]) / 2
            label = (f["description"] or f["type"])[:20]
            if width > length * 0.04:
                ax.text(mid, 0, label, ha="center", va="center",
                        fontsize=7, color="white", zorder=3)

        for f in features:
            if f["type"] in _SITE_LIKE:
                ax.plot([f["start"]], [0.55], marker="v",
                        color="#222", markersize=5, zorder=4)

        if residue and 1 <= residue <= length:
            ax.plot([residue, residue], [0, 1.3], color="crimson", lw=1.5, zorder=5)
            ax.plot([residue], [1.3], marker="o", color="crimson",
                    markersize=9, zorder=6)
            ax.text(residue, 1.45, f"p.{residue}", ha="center", va="bottom",
                    fontsize=8, color="crimson", fontweight="bold")

        ax.set_yticks([])
        ax.set_xlabel("Residue position")
        ax.set_title(f"{symbol} — UniProt domain architecture ({length} aa)",
                     fontsize=10)
        for spine in ("top", "right", "left"):
            ax.spines[spine].set_visible(False)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:  # noqa: BLE001
        return None


def _fetch_pdb_text(pdb_url: str) -> str | None:
    try:
        import httpx
        r = httpx.get(pdb_url, timeout=20.0)
        r.raise_for_status()
        return r.text
    except Exception:  # noqa: BLE001
        return None


def _parse_ca_coords(pdb_text: str) -> tuple[list[int], list[tuple[float, float, float]]]:
    """Pure-Python CA-atom parser — avoids the BioPython dependency."""
    res_ids: list[int] = []
    coords: list[tuple[float, float, float]] = []
    seen: set[int] = set()
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM"):
            continue
        if line[12:16].strip() != "CA":
            continue
        try:
            res_id = int(line[22:26])
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
        except ValueError:
            continue
        if res_id in seen:
            continue
        seen.add(res_id)
        res_ids.append(res_id)
        coords.append((x, y, z))
    return res_ids, coords


def _render_3d_structure(
    symbol: str,
    pdb_text: str,
    residue: int | None,
    features: list[dict[str, Any]],
) -> str | None:
    """Render a real 3-D backbone trace from AlphaFold PDB coordinates.

    Highlights:
      * grey backbone trace through all Cα atoms
      * affected UniProt domain in blue
      * mutated residue as a red sphere with label
      * nearby active / binding sites as orange spheres

    Returned as a base64-encoded PNG; this is fed to the VLM alongside
    the existing 2-D domain map.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    except Exception:  # noqa: BLE001
        return None

    res_ids, coords = _parse_ca_coords(pdb_text)
    if not coords:
        return None

    try:
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        zs = [c[2] for c in coords]

        fig = Figure(figsize=(7.5, 6.0), dpi=120)
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(xs, ys, zs, color="#bbbbbb", lw=1.3, zorder=1)

        # Affected domain in blue.
        domain = next(
            (f for f in features
             if f.get("type") in _DOMAIN_LIKE
             and residue is not None
             and f["start"] <= residue <= f["end"]),
            None,
        )
        if domain:
            in_dom = [
                (xs[i], ys[i], zs[i])
                for i, rid in enumerate(res_ids)
                if domain["start"] <= rid <= domain["end"]
            ]
            if len(in_dom) >= 2:
                dx, dy, dz = zip(*in_dom)
                ax.plot(dx, dy, dz, color="#4a7fd6", lw=2.4, zorder=2)

        # Active / binding sites — orange spheres.
        site_residues = {
            f["start"] for f in features
            if f.get("type") in _SITE_LIKE
        }
        for rid, (x, y, z) in zip(res_ids, coords):
            if rid in site_residues:
                ax.scatter([x], [y], [z], color="#ff7f0e",
                           s=60, edgecolors="black", linewidths=0.4, zorder=4)

        # Mutated residue.
        if residue is not None:
            try:
                idx = res_ids.index(residue)
                rx, ry, rz = coords[idx]
                ax.scatter([rx], [ry], [rz], color="crimson",
                           s=160, edgecolors="black", linewidths=0.6, zorder=5)
                ax.text(rx, ry, rz + 1.5, f"p.{residue}", color="crimson",
                        fontsize=9, fontweight="bold", zorder=6)
            except ValueError:
                pass

        ax.set_title(f"{symbol} — AlphaFold backbone (mutated residue in red)",
                     fontsize=10)
        ax.set_xlabel("x (Å)", fontsize=8)
        ax.set_ylabel("y (Å)", fontsize=8)
        ax.set_zlabel("z (Å)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(False)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:  # noqa: BLE001
        return None


def fetch_structure(mq: MutationQuery) -> dict[str, Any]:
    """Resolve UniProt + AlphaFold + domain plot for a single-gene variant."""
    if not mq.gene:
        return {"found": False, "reason": "no gene"}
    if mq.variant_class in (VariantClass.FUSION, VariantClass.CNV_AMP,
                            VariantClass.CNV_DEL, VariantClass.EXON_SKIP):
        return {"found": False, "reason": f"not applicable for {mq.variant_class}"}

    acc = _uniprot_accession(mq.gene)
    if not acc:
        return {"found": False, "reason": "no UniProt accession"}

    record = _uniprot_record(acc)
    pdb_url = _alphafold_pdb_url(acc)
    base = {
        "found": True,
        "uniprot_id": acc,
        "alphafold_pdb_url": pdb_url,
        "alphafold_viewer_url": ALPHAFOLD_VIEWER.format(acc=acc),
    }
    if not record:
        return base

    length, features = _trim_features(record)
    residue = _residue_from_query(mq)
    domain = _domain_at(residue, features) if residue and length else None
    nearby = _nearby_sites(residue, features) if residue else []
    plot_b64 = (
        _render_plot(mq.gene, length, features, residue)
        if INCLUDE_DOMAIN_PLOT and length else None
    )

    structure_3d_b64: str | None = None
    if INCLUDE_3D_PLOT and pdb_url:
        pdb_text = _fetch_pdb_text(pdb_url)
        if pdb_text:
            structure_3d_b64 = _render_3d_structure(
                mq.gene or acc, pdb_text, residue, features
            )

    protein_name = (
        ((record.get("proteinDescription") or {}).get("recommendedName") or {})
        .get("fullName", {}).get("value")
    )

    return {
        **base,
        "protein_name": protein_name,
        "protein_length": length,
        "mutated_residue": residue,
        "domain_at_residue": domain,
        "nearby_sites": nearby,
        "features": features[:MAX_FEATURES_IN_PROMPT],
        "domain_plot_png_b64": plot_b64,
        "structure_3d_png_b64": structure_3d_b64,
    }
