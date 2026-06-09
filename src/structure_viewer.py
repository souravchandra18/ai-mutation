"""Self-contained 3Dmol.js viewer HTML builder.

Extracted from `src/app.py` so both the Streamlit and Gradio frontends can
embed the same annotated AlphaFold viewer without pulling streamlit as a
dependency.
"""
from __future__ import annotations

import json as _json


def threedmol_html(
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
        viewer.setStyle({{}}, {{ cartoon: {{ color: '#d8d8d8' }} }});
        var domainOn = !!P.domain;
        if (P.domain) {{
          viewer.addStyle(
            {{ resi: P.domain.start + '-' + P.domain.end }},
            {{ cartoon: {{ color: '#4a7fd6' }} }}
          );
        }}
        var contextOn = false;
        function applyMutationAndSites() {{
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
        if (P.domain && P.domainLabel) {{
          var mid = Math.round((P.domain.start + P.domain.end) / 2);
          viewer.addLabel(P.domainLabel, {{
            position: {{ resi: mid }},
            backgroundColor: '#4a7fd6', fontColor: 'white',
            fontSize: 11, backgroundOpacity: 0.85, inFront: true
          }});
        }}
        if (P.residue) {{ viewer.zoomTo({{ resi: P.residue }}); viewer.zoom(0.65); }}
        else {{ viewer.zoomTo(); }}
        viewer.render();
        statusEl.style.display = 'none';
        legendEl.style.display = 'block';
        controlsEl.style.display = 'flex';

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
