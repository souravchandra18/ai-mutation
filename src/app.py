"""Streamlit frontend — talks to the FastAPI backend over HTTP.

Start backend:   uvicorn src.api:app --reload --port 8000
Start frontend:  streamlit run src/app.py
"""
from __future__ import annotations

import base64
import json
import os

import httpx
import streamlit as st
import streamlit.components.v1 as components
try:
    from .llm_config import DEFAULT_MODELS, get_llm_settings, is_vision_model
except ImportError:  # Streamlit can execute this file outside package context.
    from src.llm_config import DEFAULT_MODELS, get_llm_settings, is_vision_model

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
LONG_TIMEOUT = httpx.Timeout(180.0, connect=10.0)


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

    models: list[str] = []
    for item in records:
        if isinstance(item, dict) and item.get("id"):
            models.append(str(item["id"]))
        elif isinstance(item, str):
            models.append(item)
    return models


st.set_page_config(page_title="Mutation → Mechanism → Therapy", layout="wide")

st.title("🧬 Mutation → Mechanism → Therapy")
st.caption(
    "Reasons from a genomic mutation to molecular mechanism and therapeutic "
    "implications using MyGene, MyVariant, ClinVar, Open Targets, and PubMed, "
    "with an AMD/OpenAI-compatible LLM endpoint. Research use only — not medical advice."
)

with st.sidebar:
    st.subheader("Backend")
    backend_url = st.text_input("FastAPI URL", value=BACKEND_URL)
    try:
        r = httpx.get(f"{backend_url}/health", timeout=3.0)
        if r.status_code == 200:
            st.success(f"Connected · {backend_url}")
        else:
            st.warning(f"HTTP {r.status_code} from /health")
    except Exception as e:  # noqa: BLE001
        st.error(f"Cannot reach backend: {e}")

    st.subheader("LLM")
    llm_settings = get_llm_settings()
    env_model = llm_settings.model
    served_models = _fetch_llm_models(llm_settings.base_url, llm_settings.api_key)
    model_options = list(dict.fromkeys([env_model, *served_models, *DEFAULT_MODELS, "Custom…"]))
    chosen = st.selectbox(
        "Model",
        options=model_options,
        index=0,
        help="Choose the model used for the 3-stage reasoning chain.",
    )
    if chosen == "Custom…":
        model = st.text_input("Custom model id", value=env_model).strip() or env_model
    else:
        model = chosen
    is_vision = is_vision_model(model)
    st.caption(
        f"Provider: `{llm_settings.display_provider}`  \n"
        f"Endpoint: `{llm_settings.base_url}`  \n"
        f"Using: `{model}`"
        + (f"  \nLive models found: `{len(served_models)}`" if served_models else "")
        + ("  \n🖥️ Multi-modal: domain map will be sent to the mechanism stage."
           if is_vision else "")
    )

mutation = st.text_input(
    "Enter a mutation",
    value="BRAF V600E",
    help='Examples: "BRAF V600E", "TP53 R175H", "EGFR L858R", "rs113488022".',
)

with st.expander("🔬 Multimodal inputs (optional)", expanded=False):
    st.caption(
        "Attach a biomedical image (H&E patch, radiology slice, microscopy field) "
        "to be scored by BiomedCLIP / CLIP, and/or a voice note to be transcribed "
        "by Whisper. Both are sent to `/analyze_mm` and merged into the evidence."
    )
    uploaded_image = st.file_uploader(
        "Biomedical image", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
        help="Scored by BiomedCLIP if available, otherwise generic CLIP.",
    )
    uploaded_voice = st.file_uploader(
        "Voice note", type=["wav", "mp3", "m4a", "flac", "ogg"],
        help="Transcribed by Whisper. The transcript is appended to the user prompt.",
    )

col_a, col_b = st.columns([1, 1])
do_analyze = col_a.button("Analyze (full reasoning)", type="primary", disabled=not mutation.strip())
do_evidence = col_b.button("Evidence only", disabled=not mutation.strip())


def _post(path: str, payload: dict) -> dict:
    with httpx.Client(timeout=LONG_TIMEOUT) as client:
        r = client.post(f"{backend_url}{path}", json=payload)
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:  # noqa: BLE001
            detail = r.text
        raise RuntimeError(f"{r.status_code}: {detail}")
    return r.json()


def _post_multipart(path: str, fields: dict, files: dict) -> dict:
    with httpx.Client(timeout=LONG_TIMEOUT) as client:
        r = client.post(f"{backend_url}{path}", data=fields, files=files)
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except Exception:  # noqa: BLE001
            detail = r.text
        raise RuntimeError(f"{r.status_code}: {detail}")
    return r.json()


def _source_found(block: dict | None) -> bool:
    if not isinstance(block, dict) or not block:
        return False
    if block.get("found") is False:
        return False
    return any(v not in (None, "", [], {}) for k, v in block.items() if k != "error")


def _render_evidence_status(evidence: dict) -> None:
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
    st.markdown("**Evidence Sources**")
    st.caption(
        "Found: " + (", ".join(found) if found else "none")
        + "  \nMissing or not applicable: "
        + (", ".join(missing) if missing else "none")
    )


def _render_grounding_metrics(grounding: dict) -> None:
    aggregate = (grounding or {}).get("aggregate") or {}
    cgs = aggregate.get("citation_grounding_score")
    hallucination = aggregate.get("hallucination_rate")
    cols = st.columns(5)
    cols[0].metric("CGS", f"{float(cgs):.2f}" if isinstance(cgs, (int, float)) else "n/a")
    cols[1].metric("Citations", aggregate.get("total_citations", 0))
    cols[2].metric("Grounded", aggregate.get("grounded", 0))
    cols[3].metric("Fabricated", aggregate.get("fabricated", 0))
    cols[4].metric(
        "Hallucination",
        f"{float(hallucination):.2f}" if isinstance(hallucination, (int, float)) else "n/a",
    )
    if aggregate.get("off_context") or aggregate.get("unknown_source"):
        st.warning(
            "Some citations were not supported by the retrieved evidence. "
            "Unsupported citations are redacted in the displayed answer."
        )


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


def _render_at_a_glance(label: str, mutation_data: dict, evidence: dict, run: dict) -> None:
    st.markdown("**At A Glance**")
    rows = [
        ("Parsed variant", label),
        ("Variant class", mutation_data.get("variant_class") or "unknown"),
        ("Gene", mutation_data.get("gene") or "not parsed"),
        ("Clinical significance", _first_clinvar_label(evidence)),
        ("Population frequency", _gnomad_label(evidence)),
        ("Structural context", _structure_label(evidence)),
        ("ESM-2 ΔPLL", _esm2_label(evidence)),
        ("Image findings", _imaging_label(evidence)),
        ("Voice note", _speech_label(evidence)),
        ("Therapy signal", _therapy_signal_label(evidence)),
        ("Model", run.get("model") or "not reported"),
    ]
    st.table([{"Item": k, "Value": v} for k, v in rows])


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


def _render_therapy(therapy_markdown: str) -> None:
    rows = _parse_markdown_table(therapy_markdown)
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
        remaining = "\n".join(
            line for line in therapy_markdown.splitlines()
            if "|" not in line or line.strip().startswith("Caveats")
        ).strip()
        if remaining:
            st.markdown(remaining)
    else:
        st.markdown(therapy_markdown)


def _render_structure_tab(struct: dict, query: dict | None = None) -> None:
    """Render the multi-modal structural panel: domain map + AlphaFold 3D viewer."""
    query = query or {}
    gene = query.get("gene") or struct.get("uniprot_id") or ""
    protein_change = query.get("protein_change") or ""
    cols = st.columns([2, 1])
    with cols[0]:
        st.markdown(
            f"**UniProt:** [{struct.get('uniprot_id')}]"
            f"(https://www.uniprot.org/uniprotkb/{struct.get('uniprot_id')})"
            + (f"  ·  **Length:** {struct.get('protein_length')} aa"
               if struct.get("protein_length") else "")
        )
        if struct.get("protein_name"):
            st.caption(struct["protein_name"])
        if struct.get("mutated_residue"):
            dom = struct.get("domain_at_residue") or {}
            dom_label = (
                f"{dom.get('description') or dom.get('type')} "
                f"({dom.get('start')}–{dom.get('end')})"
                if dom else "no annotated domain"
            )
            st.markdown(f"**Residue p.{struct['mutated_residue']}** sits in: *{dom_label}*")
            nearby = struct.get("nearby_sites") or []
            if nearby:
                st.markdown(
                    "**Nearby active/binding sites:** "
                    + ", ".join(
                        f"{s['type']} @ {s['start']}"
                        + (f" — {s['description']}" if s.get("description") else "")
                        for s in nearby
                    )
                )

    with cols[1]:
        if struct.get("alphafold_viewer_url"):
            st.markdown(
                f"[Open in AlphaFold ↗]({struct['alphafold_viewer_url']})"
            )

    plot_b64 = struct.get("domain_plot_png_b64")
    if plot_b64:
        st.image(
            base64.b64decode(plot_b64),
            caption="UniProt domain map (red lollipop = mutated residue) — "
                    "this image is what vision-capable LLMs receive.",
            use_container_width=True,
        )

    pdb_url = struct.get("alphafold_pdb_url")
    if pdb_url:
        st.markdown("#### AlphaFold predicted 3-D structure")
        try:
            pdb_text = httpx.get(pdb_url, timeout=30.0).text
            html = _threedmol_html(
                pdb_text=pdb_text,
                residue=struct.get("mutated_residue"),
                domain=struct.get("domain_at_residue") or {},
                features=struct.get("features") or [],
                protein_change=protein_change,
                gene=gene,
            )
            components.html(html, height=560)
        except Exception as e:  # noqa: BLE001
            st.warning(
                f"3-D viewer failed: {e}. PDB available at [{pdb_url}]({pdb_url})."
            )


def _threedmol_html(
    pdb_text: str,
    residue: int | None,
    domain: dict | None = None,
    features: list[dict] | None = None,
    protein_change: str = "",
    gene: str = "",
) -> str:
    """Build a self-contained, annotated 3Dmol.js viewer.

    Annotations rendered:
      • Whole chain as light-grey cartoon
      • Affected UniProt domain in cornflower-blue
      • Mutated residue: red sphere + sticks + floating label
      • Residues within 5 Å of the mutation CA: pink lines (local context)
      • Nearby active / binding sites: orange sticks + labels
      • Legend overlay + toggle buttons (domain / context / sites / reset view)
    """
    import json as _json

    domain = domain or {}
    sites = [
        f for f in (features or [])
        if f.get("type") in ("Active site", "Binding site", "Site")
        and isinstance(f.get("start"), int)
    ]
    payload = {
        "pdb": pdb_text,
        "residue": int(residue) if residue else None,
        "domain": {
            "start": domain.get("start"),
            "end": domain.get("end"),
            "label": (domain.get("description") or domain.get("type") or ""),
        } if domain.get("start") and domain.get("end") else None,
        "sites": [
            {"resi": int(f["start"]),
             "label": (f.get("description") or f.get("type") or "site")[:40]}
            for f in sites[:8]
        ],
        "residueLabel": (
            f"{gene} {protein_change}".strip()
            or (f"{gene} p.{residue}" if residue else "")
        ),
        "domainLabel": (
            f"{(domain.get('description') or domain.get('type') or 'domain')} "
            f"({domain.get('start')}–{domain.get('end')})"
            if domain.get("start") and domain.get("end") else ""
        ),
    }
    payload_js = _json.dumps(payload)

    return f"""
<!doctype html>
<html><head><meta charset="utf-8">
<script src="https://cdn.jsdelivr.net/npm/3dmol@2.4.2/build/3Dmol-min.js"></script>
<style>
  html, body {{ margin: 0; padding: 0; background: #fafafa;
                font-family: -apple-system, system-ui, sans-serif; }}
  #wrap {{ position: relative; width: 100%; height: 540px; }}
  #viewer {{ position: absolute; inset: 0; }}
  #status {{ position: absolute; top: 8px; left: 12px; color: #888;
             font-size: 12px; z-index: 10; pointer-events: none;
             background: rgba(250,250,250,0.85); padding: 2px 6px;
             border-radius: 4px; }}
  #legend {{ position: absolute; bottom: 10px; left: 10px;
             background: rgba(255,255,255,0.92); border: 1px solid #ddd;
             border-radius: 6px; padding: 6px 10px; font-size: 11px;
             line-height: 1.55; z-index: 5; max-width: 260px; }}
  #legend b {{ font-size: 11px; }}
  .sw {{ display: inline-block; width: 10px; height: 10px; margin-right: 5px;
         border-radius: 2px; vertical-align: middle; border: 1px solid #999; }}
  #controls {{ position: absolute; top: 8px; right: 8px; z-index: 11;
               display: flex; flex-direction: column; gap: 4px;
               background: rgba(255,255,255,0.92); border: 1px solid #ddd;
               border-radius: 6px; padding: 6px; font-size: 11px; }}
  #controls button {{ font-size: 11px; padding: 3px 8px; cursor: pointer;
                       border: 1px solid #bbb; background: #f7f7f7;
                       border-radius: 4px; }}
  #controls button.active {{ background: #e0ecff; border-color: #4a7fd6; }}
</style>
</head>
<body>
<div id="wrap">
  <div id="viewer"></div>
  <div id="status">Loading 3-D structure…</div>
  <div id="legend" style="display:none">
    <b>Legend</b><br>
    <span class="sw" style="background:#d8d8d8"></span> protein chain<br>
    <span class="sw" style="background:#4a7fd6"></span> affected domain<br>
    <span class="sw" style="background:#d62728"></span> mutated residue<br>
    <span class="sw" style="background:#f4b6c2"></span> residues within 5 Å<br>
    <span class="sw" style="background:#ff7f0e"></span> active / binding site
  </div>
  <div id="controls" style="display:none">
    <button id="b-domain" class="active">Domain</button>
    <button id="b-context" class="active">5 Å context</button>
    <button id="b-sites" class="active">Sites</button>
    <button id="b-surface">Surface</button>
    <button id="b-reset">Reset view</button>
  </div>
</div>
<script>
  (function() {{
    var P = {payload_js};
    var statusEl = document.getElementById('status');
    var legendEl = document.getElementById('legend');
    var controlsEl = document.getElementById('controls');
    function fail(msg) {{
      statusEl.textContent = 'Viewer error: ' + msg;
      statusEl.style.color = '#c00';
      statusEl.style.background = 'rgba(255,235,235,0.95)';
    }}
    function probeWebGL() {{
      try {{
        var c = document.createElement('canvas');
        if (c.getContext('webgl2')) return {{ok: true}};
        if (c.getContext('webgl') || c.getContext('experimental-webgl')) return {{ok: true}};
        if (!window.WebGLRenderingContext)
          return {{ok: false, reason: 'WebGLRenderingContext unavailable in this browser build'}};
        return {{ok: false, reason: 'getContext(webgl) returned null — GPU blocklisted or hardware acceleration disabled'}};
      }} catch (e) {{ return {{ok: false, reason: String(e)}}; }}
    }}
    function init() {{
      var probe = probeWebGL();
      if (!probe.ok) {{
        fail('WebGL unavailable — ' + probe.reason +
             '. Check chrome://gpu, enable "Use graphics acceleration" in chrome://settings/system, ' +
             'or set chrome://flags/#ignore-gpu-blocklist to Enabled, then restart the browser.');
        return;
      }}
      try {{
        var wrap = document.getElementById('wrap');
        var el   = document.getElementById('viewer');
        var w = wrap.clientWidth  || 720;
        var h = wrap.clientHeight || 540;
        var viewer = $3Dmol.createViewer(el, {{ backgroundColor: '#fafafa', width: w, height: h }});
        viewer.addModel(P.pdb, 'pdb');

        // Base: light grey cartoon for the whole chain
        viewer.setStyle({{}}, {{ cartoon: {{ color: '#d8d8d8' }} }});

        // Affected domain in blue
        var domainOn = !!P.domain;
        if (P.domain) {{
          viewer.addStyle(
            {{ resi: P.domain.start + '-' + P.domain.end }},
            {{ cartoon: {{ color: '#4a7fd6' }} }}
          );
        }}

        // Local 5 Å context around the mutated residue
        var contextOn = false;
        function showContext(on) {{
          contextOn = on;
          if (!P.residue) return;
          if (on) {{
            viewer.addStyle(
              {{ within: {{ distance: 5.0, sel: {{ resi: P.residue }} }} }},
              {{ stick: {{ color: '#f4b6c2', radius: 0.12 }} }}
            );
          }} else {{
            viewer.removeAllShapes();
            viewer.setStyle({{}}, {{ cartoon: {{ color: '#d8d8d8' }} }});
            if (domainOn && P.domain) viewer.addStyle(
              {{ resi: P.domain.start + '-' + P.domain.end }},
              {{ cartoon: {{ color: '#4a7fd6' }} }});
            applyMutationAndSites();
          }}
          viewer.render();
        }}

        function applyMutationAndSites() {{
          // Mutated residue (always shown)
          if (P.residue) {{
            viewer.addStyle({{ resi: P.residue }},
              {{ stick: {{ colorscheme: 'redCarbon', radius: 0.28 }} }});
            viewer.addStyle({{ resi: P.residue }},
              {{ sphere: {{ color: '#d62728', radius: 1.6 }} }});
            if (P.residueLabel) {{
              viewer.addLabel(P.residueLabel, {{
                position: {{ resi: P.residue }},
                backgroundColor: '#d62728', fontColor: 'white',
                fontSize: 12, backgroundOpacity: 0.85,
                borderThickness: 1, borderColor: '#7a1111',
                inFront: true
              }});
            }}
          }}
          // Active / binding sites
          (P.sites || []).forEach(function(s) {{
            viewer.addStyle({{ resi: s.resi }},
              {{ stick: {{ color: '#ff7f0e', radius: 0.22 }} }});
            viewer.addLabel(s.label, {{
              position: {{ resi: s.resi }},
              backgroundColor: '#ff7f0e', fontColor: 'white',
              fontSize: 10, backgroundOpacity: 0.8,
              inFront: true
            }});
          }});
        }}
        applyMutationAndSites();

        // Domain label floating above its midpoint
        if (P.domain && P.domainLabel) {{
          var mid = Math.round((P.domain.start + P.domain.end) / 2);
          viewer.addLabel(P.domainLabel, {{
            position: {{ resi: mid }},
            backgroundColor: '#4a7fd6', fontColor: 'white',
            fontSize: 11, backgroundOpacity: 0.85, inFront: true
          }});
        }}

        // Initial framing: zoom to the mutated residue if known, else whole chain
        if (P.residue) {{ viewer.zoomTo({{ resi: P.residue }}); viewer.zoom(0.65); }}
        else {{ viewer.zoomTo(); }}

        viewer.render();
        statusEl.style.display = 'none';
        legendEl.style.display = 'block';
        controlsEl.style.display = 'flex';

        // ----- Controls -----
        var surfaceHandle = null;
        function rebuild() {{
          viewer.setStyle({{}}, {{ cartoon: {{ color: '#d8d8d8' }} }});
          if (domainOn && P.domain) viewer.addStyle(
            {{ resi: P.domain.start + '-' + P.domain.end }},
            {{ cartoon: {{ color: '#4a7fd6' }} }});
          viewer.removeAllLabels();
          if (P.domain && P.domainLabel && domainOn) {{
            var mid = Math.round((P.domain.start + P.domain.end) / 2);
            viewer.addLabel(P.domainLabel, {{
              position: {{ resi: mid }},
              backgroundColor: '#4a7fd6', fontColor: 'white',
              fontSize: 11, backgroundOpacity: 0.85, inFront: true
            }});
          }}
          applyMutationAndSites();
          if (contextOn && P.residue) {{
            viewer.addStyle(
              {{ within: {{ distance: 5.0, sel: {{ resi: P.residue }} }} }},
              {{ stick: {{ color: '#f4b6c2', radius: 0.12 }} }}
            );
          }}
          viewer.render();
        }}

        document.getElementById('b-domain').onclick = function() {{
          domainOn = !domainOn;
          this.classList.toggle('active', domainOn);
          rebuild();
        }};
        document.getElementById('b-context').onclick = function() {{
          contextOn = !contextOn;
          this.classList.toggle('active', contextOn);
          rebuild();
        }};
        document.getElementById('b-sites').onclick = function() {{
          // Toggle labels on sites by re-using applyMutationAndSites with empty list
          P._sitesHidden = !P._sitesHidden;
          this.classList.toggle('active', !P._sitesHidden);
          var saved = P.sites;
          if (P._sitesHidden) P.sites = [];
          rebuild();
          P.sites = saved;
        }};
        document.getElementById('b-surface').onclick = function() {{
          if (surfaceHandle === null) {{
            surfaceHandle = viewer.addSurface($3Dmol.SurfaceType.VDW,
              {{ opacity: 0.55, color: '#bbb' }},
              P.residue ? {{ within: {{ distance: 8.0, sel: {{ resi: P.residue }} }} }} : {{}});
            this.classList.add('active');
          }} else {{
            viewer.removeSurface(surfaceHandle);
            surfaceHandle = null;
            this.classList.remove('active');
          }}
          viewer.render();
        }};
        document.getElementById('b-reset').onclick = function() {{
          if (P.residue) {{ viewer.zoomTo({{ resi: P.residue }}); viewer.zoom(0.65); }}
          else viewer.zoomTo();
          viewer.render();
        }};

        // Click-to-label residue under cursor
        viewer.setClickable({{}}, true, function(atom) {{
          if (!atom) return;
          viewer.addLabel(
            atom.resn + atom.resi + ' (' + atom.atom + ')',
            {{ position: {{x: atom.x, y: atom.y, z: atom.z}},
               backgroundColor: '#333', fontColor: 'white',
               fontSize: 10, backgroundOpacity: 0.8, inFront: true }}
          );
          viewer.render();
        }});

        window.addEventListener('resize', function() {{ viewer.resize(); }});
      }} catch (e) {{ fail(e.message || String(e)); }}
    }}
    function whenReady() {{
      if (!window.$3Dmol) {{ return false; }}
      requestAnimationFrame(init);
      return true;
    }}
    if (document.readyState === 'complete') {{
      if (!whenReady()) {{
        var tries = 0;
        var iv = setInterval(function() {{
          if (whenReady()) {{ clearInterval(iv); }}
          else if (++tries > 50) {{
            clearInterval(iv);
            fail('3Dmol.js failed to load (CDN blocked?)');
          }}
        }}, 100);
      }}
    }} else {{
      window.addEventListener('load', function() {{
        if (!whenReady()) {{
          var tries = 0;
          var iv = setInterval(function() {{
            if (whenReady()) {{ clearInterval(iv); }}
            else if (++tries > 50) {{
              clearInterval(iv);
              fail('3Dmol.js failed to load (CDN blocked?)');
            }}
          }}, 100);
        }}
      }});
    }}
  }})();
</script>
</body></html>
"""


if do_evidence:
    with st.spinner("Gathering evidence from biomedical sources…"):
        try:
            if uploaded_image is not None or uploaded_voice is not None:
                files: dict = {}
                if uploaded_image is not None:
                    files["image"] = (uploaded_image.name, uploaded_image.getvalue(),
                                       uploaded_image.type or "application/octet-stream")
                if uploaded_voice is not None:
                    files["voice"] = (uploaded_voice.name, uploaded_voice.getvalue(),
                                       uploaded_voice.type or "application/octet-stream")
                # /analyze_mm always runs reasoning; for evidence-only multimodal
                # we still hit /evidence (text-only) plus a side-channel note.
                data = _post("/evidence", {"mutation": mutation})
                st.info(
                    "Multimodal inputs are only used in `Analyze`. "
                    "Showing text-evidence only."
                )
            else:
                data = _post("/evidence", {"mutation": mutation})
        except Exception as e:  # noqa: BLE001
            st.error(str(e))
            st.stop()
    st.subheader(f"Parsed: `{data['mutation'].get('label')}`")
    st.json(data["evidence"])

if do_analyze:
    with st.spinner(f"Calling backend (evidence + 3-stage LLM reasoning via `{model}`)…"):
        try:
            if uploaded_image is not None or uploaded_voice is not None:
                files = {}
                if uploaded_image is not None:
                    files["image"] = (uploaded_image.name, uploaded_image.getvalue(),
                                       uploaded_image.type or "application/octet-stream")
                if uploaded_voice is not None:
                    files["voice"] = (uploaded_voice.name, uploaded_voice.getvalue(),
                                       uploaded_voice.type or "application/octet-stream")
                data = _post_multipart(
                    "/analyze_mm",
                    fields={"mutation": mutation, "model": model},
                    files=files,
                )
            else:
                data = _post("/analyze", {"mutation": mutation, "model": model})
        except Exception as e:  # noqa: BLE001
            st.error(str(e))
            st.stop()

    label = data["mutation"].get("label") or mutation
    st.subheader(f"Parsed: `{label}`")

    r = data["reasoning"]
    evidence = data.get("evidence") or {}
    grounding = data.get("grounding") or {}
    run = data.get("run") or {}
    _render_at_a_glance(label, data.get("mutation") or {}, evidence, run)
    _render_grounding_metrics(grounding)

    with st.expander("Raw evidence (JSON)", expanded=False):
        st.json(evidence)

    struct = (data["evidence"] or {}).get("structure") or {}
    has_structure = bool(struct.get("found"))

    tabs = st.tabs(
        ["1. Summary", "2. Mechanism", "3. Therapy", "4. Trust & Evidence"]
        + (["🧬 Structure"] if has_structure else [])
    )
    with tabs[0]:
        st.markdown(r["mutation_summary"])
    with tabs[1]:
        st.markdown(r["mechanism"])
    with tabs[2]:
        _render_therapy(r["therapy"])
    with tabs[3]:
        _render_evidence_status(evidence)
        st.markdown("**Citation Verification**")
        _render_grounding_metrics(grounding)
        per_stage = grounding.get("per_stage") or {}
        if per_stage:
            with st.expander("Per-stage verification details", expanded=False):
                st.json(per_stage)
        if run:
            with st.expander("Run metadata", expanded=False):
                st.json(run)

    if has_structure:
        with tabs[4]:
            _render_structure_tab(struct, query=data.get("mutation") or {})

    report = (
        f"# Mutation → Mechanism → Therapy: {label}\n\n"
        f"## 1. Mutation Summary\n{r['mutation_summary']}\n\n"
        f"## 2. Molecular Mechanism\n{r['mechanism']}\n\n"
        f"## 3. Therapeutic Implications\n{r['therapy']}\n\n"
        f"## 4. Citation Grounding\n"
        f"```json\n{json.dumps(grounding.get('aggregate', {}), indent=2)}\n```\n"
    )
    st.download_button(
        "Download Markdown report",
        data=report,
        file_name=f"{label.replace(' ', '_')}_report.md",
        mime="text/markdown",
    )

