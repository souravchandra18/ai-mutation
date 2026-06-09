# Mutation → Mechanism → Therapy Reasoning

AI-driven pipeline that reasons from a **genomic mutation** to the
**molecular mechanism** it perturbs and the **therapeutic options** that
target it, grounded in public biomedical databases.

| Layer | Source |
|-------|--------|
| Gene metadata & symbol canonicalisation | [MyGene.info](https://mygene.info) |
| Variant annotation + in-silico scores | [MyVariant.info](https://myvariant.info) (dbSNP, dbNSFP, CADD, COSMIC, **REVEL**, **AlphaMissense**, gnomAD AF) |
| Clinical significance | [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/) via NCBI E-utilities |
| Target / disease / drug | [Open Targets Platform](https://platform.opentargets.org/) GraphQL |
| Curated variant-level therapy evidence | [CIViC](https://civicdb.org/) GraphQL — incl. `evidenceItems` (level, type, direction, significance, disease, therapies, source) |
| Literature | [PubMed](https://pubmed.ncbi.nlm.nih.gov/) via NCBI E-utilities (titles **and** abstracts via `efetch`) |
| Optional HGVS normalisation | [Mutalyzer 3](https://mutalyzer.nl/) REST |
| **Structural context (multi-modal)** | [UniProt](https://www.uniprot.org/) features + [AlphaFold](https://alphafold.ebi.ac.uk/) predicted 3-D structure (2-D domain map **and** 3-D backbone render) |
| **Protein language model** (multi-modal) | [ESM-2](https://huggingface.co/facebook/esm2_t33_650M_UR50D) zero-shot variant-effect score (ΔPLL) — in-process on ROCm |
| **Biomedical image encoder** (multi-modal, optional) | [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224) via `open_clip_torch`, falls back to OpenAI CLIP |
| **Speech transcription** (multi-modal, optional) | [Whisper](https://huggingface.co/openai/whisper-base) via `transformers` ASR pipeline |
| Reasoning | AMD AI Developer Cloud via vLLM/AIM or any OpenAI-compatible chat endpoint |

All independent data-source calls run **in parallel** (thread pool) inside
`evidence.gather()`, so an end-to-end evidence fetch is dominated by the
slowest single source rather than their sum.

## Multi-modal reasoning

The pipeline is genuinely multi-modal: a single `vllm serve` process hosts
the reasoning LLM, and three specialist models run alongside it (in the
FastAPI / Gradio / notebook process) on the same ROCm GPU.

| Modality | Model (where it runs) | Output fed to the LLM |
|---|---|---|
| **Text** — biomedical RAG | 7 public APIs (existing) | JSON evidence block |
| **Protein sequence** | ESM-2 650M (`facebook/esm2_t33_650M_UR50D`) in-process | `evidence.esm2.delta_pll` + classification → `[ESM2]` |
| **Structure (image)** | UniProt + AlphaFold + Matplotlib (2-D domain map and 3-D Cα backbone trace) | Two PNGs as `image_url` content parts on the mechanism stage — sent to vision-capable models |
| **Histology / radiology (optional)** | BiomedCLIP via `open_clip_torch` (or generic CLIP fallback) | Top-k findings text → `[BiomedCLIP]` |
| **Speech (optional)** | Whisper (`openai/whisper-base`) via `transformers` | Transcript appended to user prompt |

Key points:

- All specialist models are **lazy-imported**. If `transformers`/`torch`/
  `open_clip_torch` are not installed, each module returns
  `{found: false, reason: "..."}` and the pipeline runs in text-only mode.
- The mechanism stage (Stage 2) attaches **up to two images** (domain map
  + 3-D backbone) when the chosen model is vision-capable
  (`Qwen/Qwen2.5-VL-7B-Instruct`, `OpenGVLab/InternVL3-8B`, etc.). Text-only
  models still receive the structural facts as JSON.
- The Gradio UI adds **file uploaders** for image and voice inputs and
  shows a **🧬 Structure** tab with the domain map and an interactive
  AlphaFold viewer (py3Dmol).
- Skipped for variant classes where a single residue isn't meaningful
  (fusions, amplification, deletion, exon-skip).

A cell-by-cell JupyterLab walk-through is provided at
[`notebooks/multimodal_demo.ipynb`](notebooks/multimodal_demo.ipynb).

## Supported variant classes

Protein changes accept both **1-letter** (`V600E`) and **3-letter** (`p.Val600Glu`)
amino-acid codes; `Ter` is normalised to `*`.

| Class | Example input |
|---|---|
| Missense | `BRAF V600E` &nbsp; `EGFR p.L858R` &nbsp; `TP53 p.Arg175His` |
| Nonsense | `TP53 R213*` &nbsp; `TP53 p.Arg213Ter` |
| Synonymous | `BRAF V600=` &nbsp; `BRAF p.Val600=` |
| In-frame deletion / insertion / duplication | `EGFR L747_E749del` &nbsp; `EGFR D770_N771insSVD` &nbsp; `EGFR Y772_A775dup` &nbsp; `EGFR p.Asp770_Asn771insSerValAsp` |
| Frameshift | `BRCA1 K45Rfs*4` &nbsp; `BRCA1 K45fs` &nbsp; `BRCA1 p.Lys45ArgfsTer4` |
| Splice-site (bare HGVS coding) | `BRCA1 c.5074+1G>A` &nbsp; `MET c.3028+1G>T` |
| Exon skip | `MET exon14skip` &nbsp; `MET ex14skip` |
| Gene fusion | `BCR::ABL1` &nbsp; `EML4-ALK` (both partners fetched) |
| Copy-number | `HER2 amplification` &nbsp; `CDKN2A deletion` |
| HGVS (transcript-prefixed) | `NM_004333.6:c.1799T>A` |
| dbSNP rsID | `rs113488022` |
| Compound (semicolon-separated) | `BRAF V600E;K601E` &nbsp; `EGFR L858R; T790M; C797S` |
| Gene-only fallback | `TP53` |

Aliases are automatically canonicalised to HGNC symbols (`HER2` → `ERBB2`, `EZH2` stays `EZH2`, etc.).

> ⚠️ **Research aid only — not medical advice.** All LLM output is grounded
> in the structured evidence above and cites it inline, but should be
> independently verified before any clinical use.

## Setup

For a full cloud-only deployment on an AMD AI Developer Cloud GPU droplet, use
[`AMD_AI_DEVELOPER_CLOUD_SETUP.md`](AMD_AI_DEVELOPER_CLOUD_SETUP.md). That guide
runs vLLM, FastAPI, and Gradio on the droplet, with your laptop used only as
a browser.

### 1. Start an AMD-hosted LLM endpoint

The app expects an OpenAI-compatible chat API. On AMD AI Developer Cloud,
the simplest path is to launch a GPU droplet and run vLLM on it.

On the AMD Developer Cloud instance:

```bash
# Pick a model your GPU size can serve. Qwen 7B is a good first smoke test.
export MODEL_ID=Qwen/Qwen2.5-7B-Instruct
export HF_TOKEN=<your-hugging-face-token-if-needed>

sudo docker run --rm \
  --group-add=video \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device /dev/kfd \
  --device /dev/dri \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --env "HF_TOKEN=$HF_TOKEN" \
  -p 127.0.0.1:8090:8000 \
  --ipc=host \
  --entrypoint python3 \
  vllm/vllm-openai-rocm:latest \
  -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --host 0.0.0.0 \
  --port 8000
```

Check the endpoint from the droplet:

```bash
curl http://localhost:8090/v1/models
```

If your AMD account gives you an AMD AI Workbench/AIM deployment URL
instead of a raw VM, use that URL and set `AI_API_STYLE=amd_inference`
in `.env`.

### 2. Install this application

Linux/macOS:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env`:

```dotenv
AI_PROVIDER=amd
AI_API_STYLE=openai
AI_BASE_URL=http://<amd-instance-public-ip>:8090/v1
AI_API_KEY=
AI_MODEL=Qwen/Qwen2.5-7B-Instruct
AI_VISION_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
BACKEND_URL=http://localhost:8000
```

If your endpoint requires a bearer token, put it in `AI_API_KEY`. For an
AMD AI Workbench/AIM deployment that posts to `/v1/inference`, use:

```dotenv
AI_API_STYLE=amd_inference
AI_BASE_URL=https://<your-workbench-url>
AI_API_KEY=<your-amd-api-key>
```

You can change the model in `.env` (`AI_MODEL`), pass `--model` on the CLI,
or pick one from the **model dropdown** in the Gradio UI. Good AMD
Developer Cloud starting points:

- `Qwen/Qwen2.5-7B-Instruct` *(good smoke-test model)*
- `meta-llama/Llama-3.1-8B-Instruct` *(gated Hugging Face license)*
- `meta-llama/Llama-3.1-70B-Instruct` *(larger MI300-class deployment)*
- `mistralai/Mixtral-8x7B-Instruct-v0.1`
- `Qwen/Qwen2.5-VL-7B-Instruct` *(multi-modal; receives the domain map image)*

### Optional environment variables

| Variable | Default | Purpose |
|---|---|---|
| `AI_PROVIDER` | `amd` | Provider label used in logs/UI. |
| `AI_API_STYLE` | `openai` | `openai` for `/v1/chat/completions`; `amd_inference` for `/v1/inference`. |
| `AI_BASE_URL` | `http://localhost:8090/v1` | OpenAI-compatible endpoint URL. |
| `AI_API_KEY` | — | Optional bearer token for protected endpoints. |
| `AI_MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | Default reasoning model id. |
| `AI_VISION_MODEL` | `Qwen/Qwen2.5-VL-7B-Instruct` | Default model for multimodal benchmark runs. |
| `NCBI_EMAIL` / `NCBI_API_KEY` | — | Lifts NCBI rate limits (3 → 10 req/s). |
| `FETCH_PUBMED_ABSTRACTS` | `1` | Set to `0` to skip PubMed `efetch` abstract download. |
| `PUBMED_ABSTRACT_CHARS` | `1200` | Max characters of abstract kept per record. |
| `CIVIC_MAX_EVIDENCE_PER_VARIANT` | `8` | Cap on CIViC `evidenceItems` per variant. |
| `EVIDENCE_MAX_WORKERS` | `8` | Thread-pool size for parallel evidence gathering. |
| `STRUCTURE_DOMAIN_PLOT` | `1` | Set to `0` to skip generating the UniProt domain map PNG. |
| `STRUCTURE_3D_PLOT` | `1` | Set to `0` to skip the AlphaFold 3-D backbone render. |
| `STRUCTURE_MAX_FEATURES` | `25` | Cap on UniProt features kept in the evidence JSON. |
| `ESM2_MODEL` | `facebook/esm2_t33_650M_UR50D` | Protein language model id. |
| `ESM2_ENABLED` | `1` | Set to `0` to skip ESM-2 entirely. |
| `BIOMEDCLIP_MODEL` | `hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` | Preferred biomedical image encoder. |
| `CLIP_FALLBACK` | `openai/clip-vit-base-patch32` | Fallback if `open_clip_torch` isn't installed. |
| `IMAGING_TOP_K` | `5` | Number of top label matches kept per image. |
| `WHISPER_MODEL` | `openai/whisper-base` | Speech transcription model id. |
| `WHISPER_ENABLED` | `1` | Set to `0` to skip Whisper entirely. |
| `BACKEND_URL` | `http://localhost:8000` | Web UI → FastAPI URL. |
| `GRADIO_HOST` | `0.0.0.0` | Bind address for the Gradio server. |
| `GRADIO_PORT` | `8501` | TCP port for the Gradio server. |
| `UI_FRAMEWORK` | `gradio` | Set to `streamlit` to launch the legacy Streamlit UI instead. |

## Usage

### CLI

```bash
# Full analysis (3-stage LLM chain)
python -m src.cli analyze "BRAF V600E"
python -m src.cli analyze "TP53 R175H" --md report.md --json report.json
python -m src.cli analyze "rs113488022"

# Override the LLM per-call
python -m src.cli analyze "MET exon14skip" -m Qwen/Qwen2.5-7B-Instruct

# Compound query (BRAF V600E + K601E in one run)
python -m src.cli analyze "BRAF V600E;K601E"

# Multi-modal: attach a biomedical image and/or voice note
python -m src.cli analyze "BRAF V600E" --image sample_he.png
python -m src.cli analyze "EGFR L858R" --voice clinician_note.wav
python -m src.cli analyze "TP53 R175H" --image slide.png --voice note.wav

# Just dump the structured evidence (no LLM call, no API cost)
python -m src.cli evidence "EGFR L858R"
python -m src.cli evidence "BRCA1 c.5074+1G>A"
python -m src.cli evidence "BCR::ABL1"
```

### Web UI (Gradio frontend + FastAPI backend)

The primary frontend is **Gradio**, because Streamlit's websocket protocol is
not supported inside the AMD AI Developer Cloud JupyterLab proxy. A legacy
Streamlit app is still kept at `src/app.py` for local use.

Open two terminals:

```bash
# Terminal 1 — backend
uvicorn src.api:app --reload --port 8000
# OpenAPI docs at http://localhost:8000/docs

# Terminal 2 — frontend (Gradio, default)
python -m src.gradio_app
# UI at http://localhost:8501  (set BACKEND_URL env var to override)

# Or use the helper script (UI_FRAMEWORK=streamlit selects the legacy UI):
bash scripts/start_ui.sh
UI_FRAMEWORK=streamlit bash scripts/start_ui.sh
```

The Gradio UI exposes a **Settings** accordion with a backend health check, a
live **Model** dropdown (with a `Custom…` text-input escape hatch) and a
refresh button so you can switch the configured model without restarting the
backend. The result view includes an **At A Glance** table, a parsed therapy
table when the model emits Markdown table output, a **Trust & Evidence** tab
with Citation-Grounding Score (CGS), citation counts, hallucination rate,
source coverage, and run metadata, and a **🧬 Structure** tab with the domain
map and an interactive AlphaFold viewer.

### AMD cloud helper scripts

On an AMD GPU droplet, after installing dependencies and creating `.env`:

```bash
chmod +x scripts/*.sh
bash scripts/start_all_tmux.sh
```

This opens a tmux session with:

```text
Pane 1: vLLM model server
Pane 2: FastAPI backend
Pane 3: Gradio UI
```

Individual scripts are also available:

```bash
bash scripts/start_vllm.sh
bash scripts/start_backend.sh
bash scripts/start_ui.sh
```

### REST API

```bash
curl -s -X POST http://localhost:8000/analyze \
     -H 'Content-Type: application/json' \
     -d '{"mutation": "BRAF V600E"}' | jq

# With an explicit model override
curl -s -X POST http://localhost:8000/analyze \
     -H 'Content-Type: application/json' \
     -d '{"mutation": "EGFR L858R; T790M", "model": "meta/llama-3.1-405b-instruct"}' | jq

curl -s -X POST http://localhost:8000/evidence \
     -H 'Content-Type: application/json' \
     -d '{"mutation": "rs113488022"}' | jq
```

Endpoints:

| Method | Path | Body | Returns |
|---|---|---|---|
| `GET`  | `/health`      | —                                              | `{"status":"ok"}` |
| `POST` | `/evidence`    | `{"mutation": "..."}`                          | parsed query + structured evidence |
| `POST` | `/analyze`     | `{"mutation": "...", "model": "..." (opt)}`    | evidence + 3-stage reasoning + citation grounding + run metadata |
| `POST` | `/analyze_mm`  | `multipart/form-data`: `mutation`, `model` (opt), `image` file (opt), `voice` file (opt) | same as `/analyze` plus `evidence.imaging` and `evidence.speech` blocks |

## How it works

```
        ┌──────────────────────────────────────────────────────────┐
        │  parse_mutation()  →  MutationQuery (gene, p.change,     │
        │                       HGVS, rsid)                        │
        └──────────────────────────────────────────────────────────┘
                                  │
        ┌──────────────────── evidence.gather() ────────────────────┐
        │  MyGene  MyVariant  ClinVar  Open Targets  CIViC  PubMed │
        │  UniProt + AlphaFold (2-D domain map + 3-D backbone PNG) │
        │  ESM-2 (ΔPLL)   BiomedCLIP (image)   Whisper (speech)    │
        └────────────────────────────────────────────────────────
                                  │
        ┌──────────────────── reasoning.reason() ───────────────────────┐
        │  Stage 1: Mutation Summary   (variant class, impact, ESM2)│
        │  Stage 2: Molecular Mechanism (multimodal: 2 images + JSON)│
        │  Stage 3: Therapeutic Implications (drug table + caveats) │
        │  ── grounded in evidence JSON, cited inline ──            │
        └──────────────────────────────────────────────────────────┘
```

The three-stage chain forces the model to commit to a variant-level
interpretation **before** discussing mechanism, and to commit to a
mechanism **before** proposing therapies — reducing the kind of
hand-waving common in single-shot prompts.

## Project layout

```
src/
  mutation.py       # parse "BRAF V600E" / 3-letter / HGVS / rsIDs / compound
  http.py           # retrying HTTP client + tiny cache
  data_sources.py   # MyGene, MyVariant, ClinVar, Open Targets, CIViC (incl.
                    # evidenceItems), PubMed (incl. abstracts), Mutalyzer
  structure.py      # UniProt features + AlphaFold + 2-D domain map + 3-D backbone PNG
  protein_lm.py     # ESM-2 zero-shot variant-effect score (sequence modality)
  imaging.py        # BiomedCLIP / CLIP biomedical image encoder (vision modality)
  speech.py         # Whisper voice-note transcription (speech modality)
  evidence.py       # parallel aggregation of all sources + specialists into Evidence
  reasoning.py      # 3-stage OpenAI-compatible LLM chain (multi-image mechanism stage)
  verification.py   # citation-grounding verifier (research-mode novelty)
  api.py            # FastAPI backend (uvicorn src.api:app)  — incl. /analyze_mm
  app.py            # Streamlit frontend (legacy; for local use only)
  gradio_app.py     # Gradio frontend (primary; works in AMD JupyterLab)
  structure_viewer.py # 3Dmol.js HTML builder shared by both frontends
  cli.py            # `python -m src.cli ...`  (supports --model/-m/--image/--voice)
notebooks/
  multimodal_demo.ipynb  # cell-by-cell JupyterLab walk-through of the multimodal pipeline
eval/               # research-mode benchmark harness (CIViC ground truth)
tests/              # offline pytest suite (parser + verifier + classifier)
paper/              # JOSS-style manuscript (paper.md + paper.bib)
scripts/            # AMD cloud startup helpers for vLLM, FastAPI, Gradio
AMD_CLOUD_SETUP.md  # cloud-only GPU droplet runbook
```

## Research mode

For peer-reviewed evaluation, `ai-mutation` ships with three additions
that turn the application into a benchmarkable research artifact:

### 1. Citation-grounding verifier (`src/verification.py`)

The methodological contribution. Every LLM-emitted citation is parsed
out and checked against an index built from the retrieved evidence
payload. Each citation receives one of four labels — `grounded`,
`fabricated`, `off_context`, `unknown_source` — and the per-output
**Citation-Grounding Score (CGS)** and **hallucination rate** are
computed.

Programmatic use:

```python
from src.evidence import gather
from src.reasoning import reason

mq, ev = gather("BRAF V600E")
res = reason(mq, ev, deterministic=True, verify=True, redact=True)
print(res.grounding["aggregate"])
# {'total_citations': 14, 'grounded': 13, 'fabricated': 0,
#  'off_context': 1, 'citation_grounding_score': 0.929,
#  'hallucination_rate': 0.0}
```

When `redact=True`, fabricated citations in the displayed strings are
rewritten as `[UNVERIFIED:PubMed:11111111]` so a downstream reader
cannot accidentally trust them.

### 2. Deterministic reasoning mode

`reason(..., deterministic=True, seed=7)` sets `temperature=0` and
forwards a fixed seed, then logs the SHA-256 prompt hash, model name,
seed, latency, and token usage for every stage under
`ReasoningResult.run`. This is what makes published numbers
reproducible (modulo upstream provider determinism).

### 3. Evaluation harness (`eval/`)

```bash
python -m eval.dataset --n 50     #  build CIViC benchmark

# run 3 baselines
python -m eval.run_benchmark \           
    --variants eval/data/civic_benchmark.jsonl \
    --modes no_rag rag_text rag_mm \
    --out eval/results/

python -m eval.run_benchmark --variants eval/data/civic_benchmark.jsonl --modes no_rag rag_text rag_mm --out eval/results/

python -m eval.score eval/results/     # aggregate → REPORT.md
```

Headline metrics, with 95 % bootstrap CIs, are reported per mode:

* **Agreement** — fraction of variants where the predicted clinical
  class matches the curated CIViC class.
* **CGS** — fraction of citations that were grounded.
* **Hallucination rate** — fraction of fabricated identifiers.
* **Tokens / latency** — cost metrics per variant.

See [`eval/README.md`](eval/README.md) for the full methodology.

### 4. Tests + CI

```bash
pip install pytest pytest-cov ruff
pytest -q                # offline tests
```


## Security notes

- The `.env` file is gitignored. **Never commit your key.**
- If a key is ever exposed (e.g. pasted into chat), revoke it in the AMD
  portal or wherever the endpoint credential was issued, then create a new one.
- Network egress is limited to the public endpoints listed above.
