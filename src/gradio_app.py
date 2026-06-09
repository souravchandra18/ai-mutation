"""Gradio frontend — talks to the FastAPI backend over HTTP.

Designed for the AMD AI Developer Cloud JupyterLab environment, where the
Streamlit websocket protocol is not supported. The Gradio queue uses plain
HTTP / SSE and works through the JupyterLab proxy.

Start backend:   uvicorn src.api:app --reload --port 8000
Start frontend:  python -m src.gradio_app
                 (or `bash scripts/start_ui.sh`)
"""
from __future__ import annotations

import base64
import html as _html
import json
import os
from typing import Any

import gradio as gr
import httpx

try:
    from .llm_config import DEFAULT_MODELS, get_llm_settings, is_vision_model
    from .structure_viewer import threedmol_html
except ImportError:  # `python src/gradio_app.py` style execution
    from src.llm_config import DEFAULT_MODELS, get_llm_settings, is_vision_model
    from src.structure_viewer import threedmol_html

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
LONG_TIMEOUT = httpx.Timeout(180.0, connect=10.0)
OCTET = "application/octet-stream"


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------

def _models_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return f"{base}/models"
    return f"{base}/v1/models"


def _fetch_llm_models(base_url: str, api_key: str = "") -> list[str]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    try:
        r = httpx.get(_models_url(base_url), headers=headers, timeout=5.0)
        r.raise_for_status()
        data = r.json()
    except Exception:  # noqa: BLE001
        return []

    records = data.get("data") if isinstance(data, dict) else data
    if not isinstance(records, list):
        return []

    out: list[str] = []
    for item in records:
        if isinstance(item, dict) and item.get("id"):
            out.append(str(item["id"]))
        elif isinstance(item, str):
            out.append(item)
    return out


def _check_backend(backend_url: str) -> str:
    try:
        r = httpx.get(f"{backend_url}/health", timeout=3.0)
        if r.status_code == 200:
            return f"✅ Connected · `{backend_url}`"
        return f"⚠️ HTTP {r.status_code} from `{backend_url}/health`"
    except Exception as e:  # noqa: BLE001
        return f"❌ Cannot reach backend at `{backend_url}` — {e}"


def _post(backend_url: str, path: str, payload: dict) -> dict:
    with httpx.Client(timeout=LONG_TIMEOUT) as client:
        r = client.post(f"{backend_url}{path}", json=payload)
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:  # noqa: BLE001
            detail = r.text
        raise RuntimeError(f"{r.status_code}: {detail}")
    return r.json()


def _post_multipart(backend_url: str, path: str, fields: dict, files: dict) -> dict:
    with httpx.Client(timeout=LONG_TIMEOUT) as client:
        r = client.post(f"{backend_url}{path}", data=fields, files=files)
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:  # noqa: BLE001
            detail = r.text
        raise RuntimeError(f"{r.status_code}: {detail}")
    return r.json()


def _file_tuple(path: str | None) -> tuple[str, bytes, str] | None:
    if not path:
        return None
    name = os.path.basename(path)
    with open(path, "rb") as fh:
        return (name, fh.read(), OCTET)


# ---------------------------------------------------------------------------
# Display helpers (mirroring the Streamlit app's labels)
# ---------------------------------------------------------------------------

def _source_found(block: Any) -> bool:
    if not isinstance(block, dict) or not block:
        return False
    if block.get("found") is False:
        return False
    return any(v not in (None, "", [], {}) for k, v in block.items() if k != "error")


def _first_clinvar_label(evidence: dict) -> str:
    clinvar = evidence.get("clinvar") or {}
    records = clinvar.get("records") or []
    if records:
        value = records[0].get("clinical_significance")
        if value:
            return str(value)
    variant_cv = ((evidence.get("variant") or {}).get("clinvar") or {}).get("rcv") or []
    if variant_cv:
        value = variant_cv[0].get("clinical_significance")
        if value:
            return str(value)
    return "not found"


def _gnomad_label(evidence: dict) -> str:
    variant = evidence.get("variant") or {}
    for key in ("gnomad_exome_af", "gnomad_genome_af"):
        if variant.get(key) is not None:
            return f"{key.replace('_af', '').replace('_', ' ')}: {variant[key]}"
    return "not found"


def _structure_label(evidence: dict) -> str:
    struct = evidence.get("structure") or {}
    if not struct.get("found"):
        return "not available"
    domain = struct.get("domain_at_residue") or {}
    if domain:
        label = domain.get("description") or domain.get("type") or "annotated feature"
        return f"{label} ({domain.get('start')}-{domain.get('end')})"
    if struct.get("mutated_residue"):
        return f"residue {struct['mutated_residue']} outside annotated domains"
    return "UniProt/AlphaFold context available"


def _therapy_signal_label(evidence: dict) -> str:
    ot = evidence.get("opentargets") or {}
    civic = evidence.get("civic") or {}
    drug_count = ot.get("known_drugs_total") or len(ot.get("known_drugs") or [])
    civic_variants = len(civic.get("variants") or [])
    bits = []
    if drug_count:
        bits.append(f"{drug_count} Open Targets drug/candidate records")
    if civic_variants:
        bits.append(f"{civic_variants} CIViC variant records")
    return "; ".join(bits) if bits else "no direct therapy records found"


def _esm2_label(evidence: dict) -> str:
    esm = evidence.get("esm2") or {}
    if not esm.get("found"):
        return esm.get("reason") or "not run"
    return f"{esm.get('delta_pll')} ({esm.get('classification')})"


def _imaging_label(evidence: dict) -> str:
    img = evidence.get("imaging") or {}
    if not img.get("found"):
        return img.get("reason") or "none uploaded"
    return img.get("summary") or "image scored"


def _speech_label(evidence: dict) -> str:
    sp = evidence.get("speech") or {}
    if not sp.get("found"):
        return sp.get("reason") or "none uploaded"
    transcript = (sp.get("transcript") or "").strip()
    return (transcript[:80] + "…") if len(transcript) > 80 else transcript


def _at_a_glance_rows(label: str, mutation_data: dict, evidence: dict, run: dict) -> list[list[str]]:
    return [
        ["Parsed variant", label],
        ["Variant class", mutation_data.get("variant_class") or "unknown"],
        ["Gene", mutation_data.get("gene") or "not parsed"],
        ["Clinical significance", _first_clinvar_label(evidence)],
        ["Population frequency", _gnomad_label(evidence)],
        ["Structural context", _structure_label(evidence)],
        ["ESM-2 ΔPLL", _esm2_label(evidence)],
        ["Image findings", _imaging_label(evidence)],
        ["Voice note", _speech_label(evidence)],
        ["Therapy signal", _therapy_signal_label(evidence)],
        ["Model", run.get("model") or "not reported"],
    ]


def _evidence_status_md(evidence: dict) -> str:
    sources = [
        ("MyGene", evidence.get("gene")),
        ("MyVariant", evidence.get("variant")),
        ("ClinVar", evidence.get("clinvar")),
        ("Open Targets", evidence.get("opentargets")),
        ("CIViC", evidence.get("civic")),
        ("PubMed", evidence.get("pubmed")),
        ("UniProt/AlphaFold", evidence.get("structure")),
        ("ESM-2", evidence.get("esm2")),
        ("BiomedCLIP", evidence.get("imaging")),
        ("Whisper", evidence.get("speech")),
    ]
    found = [name for name, block in sources if _source_found(block)]
    missing = [name for name, block in sources if not _source_found(block)]
    return (
        "**Evidence Sources**\n\n"
        "Found: " + (", ".join(found) if found else "_none_") + "\n\n"
        "Missing or not applicable: " + (", ".join(missing) if missing else "_none_")
    )


def _grounding_md(grounding: dict) -> str:
    aggregate = (grounding or {}).get("aggregate") or {}
    cgs = aggregate.get("citation_grounding_score")
    hallucination = aggregate.get("hallucination_rate")

    def _fmt(value: Any) -> str:
        return f"{float(value):.2f}" if isinstance(value, (int, float)) else "n/a"

    rows = [
        f"| CGS | Citations | Grounded | Fabricated | Hallucination |",
        f"|---|---|---|---|---|",
        f"| **{_fmt(cgs)}** | {aggregate.get('total_citations', 0)} "
        f"| {aggregate.get('grounded', 0)} "
        f"| {aggregate.get('fabricated', 0)} "
        f"| **{_fmt(hallucination)}** |",
    ]
    note = ""
    if aggregate.get("off_context") or aggregate.get("unknown_source"):
        note = (
            "\n\n> ⚠️ Some citations were not supported by the retrieved evidence. "
            "Unsupported citations are redacted in the displayed answer."
        )
    return "\n".join(rows) + note


def _parse_markdown_table(markdown: str) -> list[dict[str, str]]:
    lines = [line.strip() for line in markdown.splitlines() if "|" in line]
    if len(lines) < 2:
        return []
    header = [cell.strip() for cell in lines[0].strip("|").split("|")]
    if not header or any(not cell for cell in header):
        return []
    rows: list[dict[str, str]] = []
    for line in lines[1:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if not cells or all(set(cell) <= {"-", ":", " "} for cell in cells):
            continue
        if len(cells) != len(header):
            continue
        rows.append(dict(zip(header, cells)))
    return rows


def _therapy_to_components(therapy_markdown: str) -> tuple[list[list[str]], list[str], str]:
    """Return (table_rows, headers, remaining_markdown)."""
    rows = _parse_markdown_table(therapy_markdown)
    if not rows:
        return [], [], therapy_markdown
    headers = list(rows[0].keys())
    table = [[r.get(h, "") for h in headers] for r in rows]
    remaining = "\n".join(
        line for line in therapy_markdown.splitlines()
        if "|" not in line or line.strip().lower().startswith("caveats")
    ).strip()
    return table, headers, remaining


def _structure_panel_md(struct: dict) -> str:
    if not struct.get("found"):
        return "_No structural context (UniProt / AlphaFold) available for this variant._"
    parts = [
        f"**UniProt:** [{struct.get('uniprot_id')}]"
        f"(https://www.uniprot.org/uniprotkb/{struct.get('uniprot_id')})",
    ]
    if struct.get("protein_length"):
        parts.append(f"**Length:** {struct['protein_length']} aa")
    if struct.get("protein_name"):
        parts.append(f"_{struct['protein_name']}_")
    if struct.get("mutated_residue"):
        dom = struct.get("domain_at_residue") or {}
        dom_label = (
            f"{dom.get('description') or dom.get('type')} "
            f"({dom.get('start')}–{dom.get('end')})"
            if dom else "no annotated domain"
        )
        parts.append(f"**Residue p.{struct['mutated_residue']}** sits in: *{dom_label}*")
    nearby = struct.get("nearby_sites") or []
    if nearby:
        parts.append(
            "**Nearby active/binding sites:** " + ", ".join(
                f"{s['type']} @ {s['start']}"
                + (f" — {s['description']}" if s.get("description") else "")
                for s in nearby
            )
        )
    if struct.get("alphafold_viewer_url"):
        parts.append(f"[Open in AlphaFold ↗]({struct['alphafold_viewer_url']})")
    return "  \n".join(parts)


def _structure_3d_iframe(struct: dict, query: dict | None = None) -> str:
    pdb_url = struct.get("alphafold_pdb_url")
    if not pdb_url:
        return ""
    query = query or {}
    try:
        pdb_text = httpx.get(pdb_url, timeout=30.0).text
    except Exception as e:  # noqa: BLE001
        return (
            f"<p style='color:#c00'>3-D viewer failed to fetch PDB: "
            f"{_html.escape(str(e))}. PDB available at "
            f"<a href='{_html.escape(pdb_url)}' target='_blank'>{_html.escape(pdb_url)}</a>.</p>"
        )
    inner = threedmol_html(
        pdb_text=pdb_text,
        residue=struct.get("mutated_residue"),
        domain=struct.get("domain_at_residue") or {},
        features=struct.get("features") or [],
        protein_change=query.get("protein_change") or "",
        gene=query.get("gene") or "",
    )
    # Embed in an iframe via srcdoc so the inline <script> tags execute in
    # an isolated browsing context (Gradio's gr.HTML strips top-level scripts).
    escaped = _html.escape(inner, quote=True)
    return (
        f'<iframe srcdoc="{escaped}" '
        f'style="width:100%;height:560px;border:1px solid #ddd;border-radius:6px;" '
        f'sandbox="allow-scripts allow-same-origin"></iframe>'
    )


def _domain_plot_html(struct: dict) -> str:
    plot_b64 = struct.get("domain_plot_png_b64")
    if not plot_b64:
        return ""
    return (
        f'<img src="data:image/png;base64,{plot_b64}" '
        f'alt="UniProt domain map" '
        f'style="max-width:100%;border:1px solid #eee;border-radius:6px;" /><br>'
        f'<small>UniProt domain map (red lollipop = mutated residue) — '
        f'this image is what vision-capable LLMs receive.</small>'
    )


def _build_report_md(label: str, r: dict, grounding: dict) -> str:
    return (
        f"# Mutation → Mechanism → Therapy: {label}\n\n"
        f"## 1. Mutation Summary\n{r.get('mutation_summary', '')}\n\n"
        f"## 2. Molecular Mechanism\n{r.get('mechanism', '')}\n\n"
        f"## 3. Therapeutic Implications\n{r.get('therapy', '')}\n\n"
        f"## 4. Citation Grounding\n"
        f"```json\n{json.dumps(grounding.get('aggregate', {}), indent=2)}\n```\n"
    )


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------

EMPTY_ANALYZE: tuple = (
    "_Run an analysis to see results._",  # status
    [],                                     # at-a-glance table
    "",                                     # grounding md
    "_Run an analysis._",                   # evidence status md
    {},                                     # raw evidence json
    "",                                     # summary md
    "",                                     # mechanism md
    [],                                     # therapy table
    [],                                     # therapy headers (state)
    "",                                     # therapy remaining md
    "",                                     # structure md
    "",                                     # domain plot html
    "",                                     # 3d viewer html
    None,                                   # downloadable report file
)


def on_check_backend(backend_url: str) -> str:
    return _check_backend(backend_url)


def on_refresh_models(backend_url: str) -> dict:
    settings = get_llm_settings()
    served = _fetch_llm_models(settings.base_url, settings.api_key)
    options = list(dict.fromkeys([settings.model, *served, *DEFAULT_MODELS, "Custom…"]))
    return gr.update(choices=options, value=settings.model)


def on_evidence(backend_url: str, mutation: str) -> tuple:
    if not mutation.strip():
        raise gr.Error("Enter a mutation first.")
    data = _post(backend_url, "/evidence", {"mutation": mutation})
    label = data["mutation"].get("label") or mutation
    return f"**Parsed:** `{label}`", data.get("evidence") or {}


def _make_report_file(label: str, report_md: str) -> str:
    out_dir = "/tmp/ai_mutation_reports"
    os.makedirs(out_dir, exist_ok=True)
    safe = label.replace(" ", "_").replace("/", "_") or "report"
    path = os.path.join(out_dir, f"{safe}_report.md")
    with open(path, "w") as fh:
        fh.write(report_md)
    return path


def on_analyze(
    backend_url: str,
    mutation: str,
    model: str,
    custom_model: str,
    image_path: str | None,
    voice_path: str | None,
) -> tuple:
    if not mutation.strip():
        raise gr.Error("Enter a mutation first.")
    chosen_model = (custom_model.strip() if model == "Custom…" else model) or get_llm_settings().model

    files: dict = {}
    img_tuple = _file_tuple(image_path)
    if img_tuple is not None:
        files["image"] = img_tuple
    voice_tuple = _file_tuple(voice_path)
    if voice_tuple is not None:
        files["voice"] = voice_tuple

    if files:
        data = _post_multipart(
            backend_url, "/analyze_mm",
            fields={"mutation": mutation, "model": chosen_model},
            files=files,
        )
    else:
        data = _post(backend_url, "/analyze", {"mutation": mutation, "model": chosen_model})

    label = data["mutation"].get("label") or mutation
    r = data.get("reasoning") or {}
    evidence = data.get("evidence") or {}
    grounding = data.get("grounding") or {}
    run = data.get("run") or {}
    struct = evidence.get("structure") or {}

    table, headers, remaining = _therapy_to_components(r.get("therapy", ""))
    therapy_table = gr.update(value=table, headers=headers or ["Therapy"]) if table else gr.update(value=[], headers=["Therapy"])

    report_md = _build_report_md(label, r, grounding)
    report_file = _make_report_file(label, report_md)

    return (
        f"### Parsed: `{label}`",
        _at_a_glance_rows(label, data.get("mutation") or {}, evidence, run),
        _grounding_md(grounding),
        _evidence_status_md(evidence),
        evidence,
        r.get("mutation_summary", ""),
        r.get("mechanism", ""),
        therapy_table,
        remaining,
        _structure_panel_md(struct),
        _domain_plot_html(struct),
        _structure_3d_iframe(struct, query=data.get("mutation") or {}),
        report_file,
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

def build_ui() -> gr.Blocks:
    settings = get_llm_settings()
    initial_models = _fetch_llm_models(settings.base_url, settings.api_key)
    model_choices = list(dict.fromkeys(
        [settings.model, *initial_models, *DEFAULT_MODELS, "Custom…"]
    ))
    is_vision = is_vision_model(settings.model)
    vision_note = (
        "  \n🖥️ Multi-modal: domain map and 3-D structure will be sent to the mechanism stage."
        if is_vision else ""
    )

    with gr.Blocks(title="Mutation → Mechanism → Therapy", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🧬 Mutation → Mechanism → Therapy\n"
            "Reasons from a genomic mutation to molecular mechanism and therapeutic "
            "implications using MyGene, MyVariant, ClinVar, Open Targets, CIViC and "
            "PubMed, fused with ESM-2, BiomedCLIP, Whisper and an AMD/OpenAI-compatible "
            "vLLM endpoint. **Research use only — not medical advice.**"
        )

        with gr.Accordion("⚙️ Settings", open=False):
            with gr.Row():
                backend_url = gr.Textbox(
                    label="FastAPI backend URL", value=BACKEND_URL, scale=3,
                )
                check_btn = gr.Button("Check", scale=1)
            backend_status = gr.Markdown(_check_backend(BACKEND_URL))

            with gr.Row():
                model_dd = gr.Dropdown(
                    label="LLM model",
                    choices=model_choices,
                    value=settings.model,
                    allow_custom_value=False,
                    scale=3,
                )
                refresh_btn = gr.Button("🔄 Refresh", scale=1)
            custom_model = gr.Textbox(
                label="Custom model id (used when 'Custom…' is selected)",
                value="", visible=False,
            )
            gr.Markdown(
                f"Provider: `{settings.display_provider}`  \n"
                f"Endpoint: `{settings.base_url}`"
                + vision_note
            )

        mutation = gr.Textbox(
            label="Enter a mutation",
            value="BRAF V600E",
            placeholder='Examples: "BRAF V600E", "TP53 R175H", "EGFR L858R", "rs113488022".',
        )

        with gr.Accordion("🔬 Multimodal inputs (optional)", open=False):
            gr.Markdown(
                "Attach a biomedical image (H&E patch, radiology slice, microscopy "
                "field) to be scored by BiomedCLIP / CLIP, and/or a voice note to be "
                "transcribed by Whisper. Both are sent to `/analyze_mm` and merged "
                "into the evidence."
            )
            with gr.Row():
                image_in = gr.File(
                    label="Biomedical image",
                    file_types=[".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"],
                    type="filepath",
                )
                voice_in = gr.File(
                    label="Voice note",
                    file_types=[".wav", ".mp3", ".m4a", ".flac", ".ogg"],
                    type="filepath",
                )

        with gr.Row():
            analyze_btn = gr.Button("Analyze (full reasoning)", variant="primary")
            evidence_btn = gr.Button("Evidence only")

        # ------- evidence-only output -------
        evidence_header = gr.Markdown("")
        evidence_json = gr.JSON(label="Evidence", value=None)

        # ------- analyze output -------
        result_header = gr.Markdown("")
        with gr.Tabs():
            with gr.Tab("📋 At a Glance"):
                glance = gr.Dataframe(
                    headers=["Item", "Value"],
                    datatype=["str", "str"],
                    row_count=(11, "fixed"),
                    interactive=False,
                    wrap=True,
                )
                grounding_md = gr.Markdown()
            with gr.Tab("1. Summary"):
                summary_md = gr.Markdown()
            with gr.Tab("2. Mechanism"):
                mechanism_md = gr.Markdown()
            with gr.Tab("3. Therapy"):
                therapy_table = gr.Dataframe(
                    headers=["Therapy"], interactive=False, wrap=True,
                )
                therapy_remaining = gr.Markdown()
            with gr.Tab("4. Trust & Evidence"):
                evidence_status_md = gr.Markdown()
                evidence_full_json = gr.JSON(label="Raw evidence (JSON)", value=None)
            with gr.Tab("🧬 Structure"):
                structure_md = gr.Markdown()
                domain_plot_html = gr.HTML()
                structure_3d_html = gr.HTML()

        report_file = gr.File(label="📥 Markdown report", interactive=False)

        # ------------------------------------------------------------------
        # Events
        # ------------------------------------------------------------------
        check_btn.click(on_check_backend, [backend_url], [backend_status])
        refresh_btn.click(on_refresh_models, [backend_url], [model_dd])

        def _toggle_custom(choice: str):
            return gr.update(visible=(choice == "Custom…"))

        model_dd.change(_toggle_custom, [model_dd], [custom_model])

        evidence_btn.click(
            on_evidence,
            inputs=[backend_url, mutation],
            outputs=[evidence_header, evidence_json],
        )

        analyze_btn.click(
            on_analyze,
            inputs=[backend_url, mutation, model_dd, custom_model, image_in, voice_in],
            outputs=[
                result_header,
                glance,
                grounding_md,
                evidence_status_md,
                evidence_full_json,
                summary_md,
                mechanism_md,
                therapy_table,
                therapy_remaining,
                structure_md,
                domain_plot_html,
                structure_3d_html,
                report_file,
            ],
        )

    return demo


def main() -> None:
    host = os.getenv("GRADIO_HOST", os.getenv("STREAMLIT_HOST", "0.0.0.0"))
    port = int(os.getenv("GRADIO_PORT", os.getenv("STREAMLIT_PORT", "8501")))
    root_path = os.getenv("GRADIO_ROOT_PATH", "")
    demo = build_ui()
    demo.queue().launch(
        server_name=host,
        server_port=port,
        root_path=root_path or None,
        show_error=True,
        share=os.getenv("GRADIO_SHARE", "0") == "1",
    )


if __name__ == "__main__":
    main()
