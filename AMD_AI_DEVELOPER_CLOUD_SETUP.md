# AMD AI Developer Cloud Setup

This guide runs the full `ai-mutation` application on AMD AI Developer Cloud.
Nothing runs on your local machine except your SSH client and web browser.

The cloud droplet will run:

```text
AMD GPU Droplet
  ├─ vLLM model server      http://localhost:8090/v1   (one reasoning model)
  ├─ FastAPI backend        http://localhost:8000
  │    └─ in-process specialist models (lazy-loaded on first call):
  │         ESM-2          — protein language model (sequence modality)
  │         BiomedCLIP/CLIP— biomedical image encoder (vision modality)
  │         Whisper        — speech transcription (speech modality)
  └─ Gradio web UI          http://0.0.0.0:8501

Your laptop
  └─ Browser opens          http://<droplet-public-ip>:8501
```

## 1. What You Need Before Starting

You need:

- AMD AI Developer Cloud account access.
- A GPU droplet with an AMD Instinct GPU.
- SSH key configured in your AMD cloud account.
- The `ai-mutation` project code available to copy onto the droplet.
- Basic terminal access.

Recommended first model:

```text
Qwen/Qwen2.5-7B-Instruct
```

This is a practical first-run model because it is smaller than 70B-class
models and does not usually require accepting a gated license.

## 2. Create A GPU Droplet

In AMD AI Developer Cloud:

1. Go to the droplet creation page.
2. Choose **GPU Droplet**.
3. Select an AMD Instinct GPU option.
4. Choose an Ubuntu/ROCm-ready image if the portal gives image choices.
5. Select or add your SSH key.
6. Create the droplet.
7. Wait until the droplet status is running.
8. Copy the droplet public IP address.

You should have:

```text
Droplet public IP: <droplet-public-ip>
SSH username: <user>
```

The username is shown in the AMD cloud portal. Common examples are:

```text
ubuntu
amd
root
```

Use the username shown by your droplet page.

## 3. Configure Firewall / Networking

Open these inbound ports for the droplet:

```text
22    SSH
8501  Gradio web UI
```

Do not expose these publicly unless you specifically need to:

```text
8000  FastAPI backend
8090  vLLM model server
```

In this guide, `8000` and `8090` stay private on the droplet.

## 4. SSH Into The Droplet

From your local machine:

```bash
ssh <user>@<droplet-public-ip>
```

Example:

```bash
ssh ubuntu@12.34.56.78
```

All remaining commands are run inside the AMD GPU droplet unless stated
otherwise.

## 4A. If You Are Using AMD JupyterLab Instead Of SSH

If your AMD AI Developer Cloud droplet opens directly into JupyterLab, you can
still run the full application from there.

Use this rule:

```text
Use JupyterLab Terminal for long-running services.
Use notebook cells only for quick checks.
```

Do not start vLLM, FastAPI, or the Gradio UI from ordinary notebook cells unless you
use background commands. A notebook cell stays busy while the server is running.
The JupyterLab **Terminal** is the cleaner option.

### Open A Terminal In JupyterLab

In JupyterLab:

```text
File -> New -> Terminal
```

All setup commands from this guide can be run in that terminal.

### Recommended JupyterLab Flow

Inside the JupyterLab Terminal:

```bash
cd ~
```

Then continue with:

```text
Section 5: Check GPU And Docker
Section 6: Install Basic System Packages
Section 7: Copy The Project To The Droplet
Section 8: Create Python Environment
Section 9: Create The .env File
Section 10: Make Startup Scripts Executable
Section 11: Start All Services
```

After the services start, open the Gradio UI at:

```text
http://<droplet-public-ip>:8501
```

If AMD JupyterLab gives you a built-in port/proxy view, expose or open port:

```text
8501
```

Keep these private/local inside the droplet:

```text
8000  FastAPI backend
8090  vLLM model server
```

### Quick Notebook Test Cell

After vLLM and FastAPI are running, you may test from a notebook cell:

```python
import requests

print(requests.get("http://localhost:8090/v1/models").json())
print(requests.get("http://localhost:8000/health").json())
```

To test analysis from a notebook cell:

```python
import requests

payload = {
    "mutation": "BRAF V600E",
    "model": "Qwen/Qwen2.5-7B-Instruct",
}

r = requests.post("http://localhost:8000/analyze", json=payload, timeout=300)
r.raise_for_status()
data = r.json()

print(data["mutation"]["label"])
print(data["grounding"]["aggregate"])
print(data["reasoning"]["mutation_summary"][:1000])
```

The notebook test is optional. The main user interface is still the Gradio app
at `src/gradio_app.py`. Gradio is used instead of Streamlit because the AMD AI
Developer Cloud JupyterLab proxy does not support Streamlit's websocket
protocol — Gradio uses plain HTTP / SSE and works through the proxy.

### Multimodal Demo Notebook

A cell-by-cell walk-through of the multimodal pipeline (ESM-2 + AlphaFold
3-D render + optional BiomedCLIP + optional Whisper, fused by the vLLM
reasoning model) is provided at:

```text
notebooks/multimodal_demo.ipynb
```

Open it in JupyterLab (`File → Open from Path → notebooks/multimodal_demo.ipynb`)
and run cells top-to-bottom. The notebook checks GPU/vLLM availability, runs
each specialist independently, and finally calls the full multimodal
pipeline end-to-end on a single variant.

## 5. Check GPU And Runtime

Check that the AMD GPU is visible:

```bash
rocm-smi
```

You should see GPU information.

Check Docker:

```bash
docker --version
```

If Docker works, the vLLM startup script can use Docker.

If Docker is missing, or if Docker exists but the daemon is not running, that is
also okay for AMD JupyterLab. The script can fall back to Python vLLM mode.

To force Python vLLM mode, set this in `.env` later:

```dotenv
VLLM_BACKEND=python
```

## 6. Install Basic System Packages

Run:

```bash
sudo apt update
sudo apt install -y git curl tmux python3 python3-venv python3-pip
```

`tmux` is recommended because it lets the servers keep running even if your
SSH session disconnects.

## 7. Copy The Project To The Droplet

Choose one option.

### Option A: Clone From Git

If your project is in a Git repository:

```bash
git clone <your-repo-url> ai-mutation
cd ai-mutation
```

### Option B: Upload A Zip Or Folder

From your local machine, create a zip of the project and copy it:

```bash
scp ai-mutation.zip <user>@<droplet-public-ip>:~
```

Then on the droplet:

```bash
sudo apt install -y unzip
unzip ai-mutation.zip
cd ai-mutation
```

Make sure you are inside the project folder:

```bash
pwd
ls
```

You should see files like:

```text
README.md
requirements.txt
src
tests
eval
```

## 8. Create Python Environment

Inside the `ai-mutation` folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Do not use `pip install --ignore-installed -r requirements.txt` for the app
environment. It can upgrade shared system packages such as FastAPI, Starlette,
protobuf, and cachetools past the versions expected by preinstalled AMD/vLLM
packages.

If Docker is not available in your AMD JupyterLab environment, also install
vLLM into the virtual environment:

```bash
python -m pip install vllm
```

If installation finishes without errors, continue.

## 8A. Optional: Install Multimodal Specialist Models

The app can run additional models alongside the vLLM reasoning LLM to make
the pipeline genuinely multimodal:

```text
ESM-2          protein language model         (sequence modality)
BiomedCLIP/CLIP biomedical image encoder      (vision modality)
Whisper        speech transcription           (speech modality)
Matplotlib 3-D AlphaFold backbone render      (structural image modality)
```

All specialists are **lazy-imported**. If you skip this section the app
still runs in text-only mode and each specialist returns
`{"found": false, "reason": "..."}`.

Inside the `ai-mutation` folder, with the venv activated:

```bash
source .venv/bin/activate
pip install transformers Pillow
```

`torch` should already be the ROCm wheel preinstalled in the AMD AI
Developer Cloud JupyterLab image. **Do not** `pip install torch` over it,
because that pulls a CUDA-only wheel and breaks ROCm.

Verify torch sees the GPU:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

Optional: install the BiomedCLIP backend (preferred over the generic CLIP
fallback):

```bash
pip install open_clip_torch
```

Optional: install audio decoding helpers if your voice clips are not
already PCM `.wav`:

```bash
sudo apt install -y ffmpeg
pip install soundfile librosa
```

First-call model downloads (cached under `~/.cache/huggingface/`):

```text
facebook/esm2_t33_650M_UR50D                       ~ 2.5 GB
openai/whisper-base                                ~ 0.15 GB
microsoft/BiomedCLIP-PubMedBERT_256-vit_base_p16_224 ~ 0.4 GB  (open_clip)
openai/clip-vit-base-patch32                       ~ 0.6 GB  (fallback)
```

VRAM accounting on the shared GPU is handled by giving vLLM less of the
card so the specialists fit. In a JupyterLab terminal, start vLLM with:

```bash
MODEL_ID=Qwen/Qwen2.5-7B-Instruct VLLM_HOST_PORT=8090 \
  VLLM_ARGS="--gpu-memory-utilization 0.65 --max-model-len 8192" \
  bash scripts/start_vllm.sh
```

That leaves ~30% of the GPU free — plenty for ESM-2 + BiomedCLIP +
Whisper combined.

## 9. Create The `.env` File

Inside the `ai-mutation` folder:

```bash
cp .env.example .env
nano .env
```

Use this content:

```dotenv
AI_PROVIDER=amd
AI_API_STYLE=openai
AI_BASE_URL=http://localhost:8090/v1
AI_API_KEY=
AI_MODEL=Qwen/Qwen2.5-7B-Instruct
AI_VISION_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
VLLM_BACKEND=auto

NCBI_EMAIL=you@example.com
NCBI_API_KEY=

BACKEND_URL=http://localhost:8000

# --- Multi-modal specialist models (optional) ---
ESM2_MODEL=facebook/esm2_t33_650M_UR50D
ESM2_ENABLED=1
BIOMEDCLIP_MODEL=hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
CLIP_FALLBACK=openai/clip-vit-base-patch32
IMAGING_TOP_K=5
WHISPER_MODEL=openai/whisper-base
WHISPER_ENABLED=1
STRUCTURE_DOMAIN_PLOT=1
STRUCTURE_3D_PLOT=1
```

Save and exit:

```text
Ctrl+O
Enter
Ctrl+X
```

For the first run, `AI_API_KEY` can stay empty because the vLLM server is only
listening on the droplet itself.

## 10. Make Startup Scripts Executable

Inside the `ai-mutation` folder:

```bash
chmod +x scripts/*.sh
```

The project includes these helper scripts:

```text
scripts/start_vllm.sh       starts the AMD ROCm vLLM model server
scripts/start_backend.sh    starts the FastAPI backend
scripts/start_ui.sh         starts the Gradio UI (set UI_FRAMEWORK=streamlit for legacy)
scripts/start_all_tmux.sh   starts all three services in tmux
```

## 11. Start All Services

Run:

```bash
bash scripts/start_all_tmux.sh
```

This creates a tmux session named:

```text
ai-mutation
```

The tmux session has three panes:

```text
Pane 1: vLLM model server
Pane 2: FastAPI backend
Pane 3: Gradio frontend
```

The first vLLM run can take several minutes because Docker may pull the image
and the model may download.

Useful tmux commands:

```text
Move between panes:  Ctrl+B then arrow key
Detach session:      Ctrl+B then D
Reattach later:      tmux attach -t ai-mutation
Stop a service:      Ctrl+C in that service pane
```

## 12. Test The vLLM Server

In any tmux pane or a new SSH terminal, run:

```bash
curl http://localhost:8090/v1/models
```

You should see a JSON response containing the model name.

If this fails, do not start the app yet. Fix vLLM first.

## 13. Test The FastAPI Backend

Run:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status":"ok"}
```

## 14. Confirm The Gradio UI Is Listening

Run:

```bash
curl http://localhost:8501
```

You should receive HTML output. If you do, the UI is running.

## 15. Open The Application

From your laptop browser, open:

```text
http://<droplet-public-ip>:8501
```

Example:

```text
http://12.34.56.78:8501
```

In the app, enter:

```text
BRAF V600E
```

Click:

```text
Analyze
```

## 16. Daily Startup Flow

After the first setup, your normal flow is:

```bash
ssh <user>@<droplet-public-ip>
cd ~/ai-mutation
bash scripts/start_all_tmux.sh
```

Open:

```text
http://<droplet-public-ip>:8501
```

## 17. Daily Shutdown Flow

Stop each running service with:

```text
Ctrl+C
```

Exit tmux:

```bash
exit
```

If you no longer need the cloud machine, stop or destroy the GPU droplet from
the AMD cloud portal to avoid using credits.

## 18. Troubleshooting

### Browser Cannot Open The Gradio UI

Check that the Gradio server is running:

```bash
curl http://localhost:8501
```

Check that port `8501` is open in the droplet firewall/security settings.

Make sure the Gradio server was started with:

```bash
GRADIO_HOST=0.0.0.0
```

### Backend Not Connected In The Web UI

Check FastAPI:

```bash
curl http://localhost:8000/health
```

Check `.env`:

```dotenv
BACKEND_URL=http://localhost:8000
```

Restart the UI after changing `.env`.

### LLM Call Fails

Check vLLM:

```bash
curl http://localhost:8090/v1/models
```

Check `.env`:

```dotenv
AI_BASE_URL=http://localhost:8090/v1
AI_MODEL=Qwen/Qwen2.5-7B-Instruct
```

The value of `AI_MODEL` should match the model served by vLLM.

### vLLM Fails With `PYDANTIC_V2` / FastAPI ImportError

If Python vLLM mode prints an error like:

```text
ImportError: cannot import name 'PYDANTIC_V2' from 'fastapi._compat'
```

the Python environment used by the `vllm` command has an inconsistent FastAPI
install. This usually happens after installing app dependencies into the
global Python environment with `pip --ignore-installed`.

Repair the package set, then rerun `bash scripts/start_vllm.sh`:

```bash
python3 -m pip uninstall -y fastapi starlette pydantic pydantic-core websockets
python3 -m pip install --upgrade --force-reinstall \
  'fastapi>=0.111,<0.116' 'starlette>=0.37.2,<1.0' \
  'pydantic>=2.7,<=2.12.3' 'websockets>=13,<16'
```

If the same import error persists, stale files were left under
`/usr/local/lib/python3.12/dist-packages`. Remove only these package folders and
metadata, then reinstall:

```bash
rm -rf /usr/local/lib/python3.12/dist-packages/fastapi \
  /usr/local/lib/python3.12/dist-packages/fastapi-*.dist-info \
  /usr/local/lib/python3.12/dist-packages/starlette \
  /usr/local/lib/python3.12/dist-packages/starlette-*.dist-info \
  /usr/local/lib/python3.12/dist-packages/pydantic \
  /usr/local/lib/python3.12/dist-packages/pydantic-*.dist-info \
  /usr/local/lib/python3.12/dist-packages/pydantic_core \
  /usr/local/lib/python3.12/dist-packages/pydantic_core-*.dist-info \
  /usr/local/lib/python3.12/dist-packages/websockets \
  /usr/local/lib/python3.12/dist-packages/websockets-*.dist-info
python3 -m pip install --upgrade --force-reinstall \
  'fastapi>=0.111,<0.116' 'starlette>=0.37.2,<1.0' \
  'pydantic>=2.7,<=2.12.3' 'websockets>=13,<16'
```

If the image blocks writes to system Python packages, use:

```bash
python3 -m pip install --break-system-packages --upgrade --force-reinstall \
  'fastapi>=0.111,<0.116' 'starlette>=0.37.2,<1.0' \
  'pydantic>=2.7,<=2.12.3' 'websockets>=13,<16'
```

### Docker Cannot Access GPU

Check:

```bash
rocm-smi
ls /dev/kfd
ls /dev/dri
```

If these do not work, the droplet image may not be ROCm-ready or the GPU was
not attached correctly.

### Model Download Requires Login

For gated Hugging Face models, set:

```bash
export HF_TOKEN=<your-token>
```

Then add this line to the Docker command:

```bash
--env "HF_TOKEN=$HF_TOKEN"
```

For the first run, prefer:

```text
Qwen/Qwen2.5-7B-Instruct
```

## 19. Recommended First Run Checklist

Use this checklist:

```text
[ ] GPU droplet is running
[ ] Port 22 is open
[ ] Port 8501 is open
[ ] SSH works
[ ] rocm-smi works
[ ] docker --version works
[ ] Project folder exists on droplet
[ ] Python venv created
[ ] requirements.txt installed
[ ] (optional) transformers + Pillow installed for multimodal specialists
[ ] (optional) open_clip_torch installed for BiomedCLIP backend
[ ] .env configured
[ ] scripts/*.sh made executable
[ ] vLLM is running on localhost:8090
[ ] curl localhost:8090/v1/models works
[ ] FastAPI is running on localhost:8000
[ ] curl localhost:8000/health works
[ ] Gradio UI is running on 0.0.0.0:8501
[ ] Browser opens http://<droplet-public-ip>:8501
[ ] BRAF V600E analysis runs
[ ] (optional) notebooks/multimodal_demo.ipynb runs end-to-end
```

## 20. Change The Model Later

The model used by the app has two parts:

```text
1. vLLM must be running the model on the AMD GPU droplet.
2. The app must send requests using the same model id.
```

If you change only the UI dropdown but vLLM is not serving that model, the
analysis will fail. So the safest process is:

```text
Stop vLLM
Start vLLM with the new model
Update .env AI_MODEL
Restart the Gradio UI
Select the model in the UI
```

### Step 1: Choose A Model

Good first options:

```text
Qwen/Qwen2.5-7B-Instruct
meta-llama/Llama-3.1-8B-Instruct
mistralai/Mistral-7B-Instruct-v0.3
```

Larger models need more GPU memory:

```text
meta-llama/Llama-3.1-70B-Instruct
mistralai/Mixtral-8x7B-Instruct-v0.1
```

Vision-capable models are useful for the structure/domain-map feature:

```text
Qwen/Qwen2.5-VL-7B-Instruct
Qwen/Qwen2-VL-7B-Instruct
```

For your first successful run, use:

```text
Qwen/Qwen2.5-7B-Instruct
```

### Step 2: Stop The Current vLLM Server

Reattach to tmux:

```bash
tmux attach -t ai-mutation
```

Move to the pane where vLLM is running.

Stop it:

```text
Ctrl+C
```

### Step 3: Start vLLM With The New Model

Example for Qwen 7B:

```bash
export MODEL_ID=Qwen/Qwen2.5-7B-Instruct
```

Example for Llama 8B:

```bash
export MODEL_ID=meta-llama/Llama-3.1-8B-Instruct
```

If the model needs Hugging Face access, set:

```bash
export HF_TOKEN=<your-hugging-face-token>
```

Then start vLLM:

```bash
cd ~/ai-mutation
bash scripts/start_vllm.sh
```

If you want to start a model without editing `.env`, run:

```bash
cd ~/ai-mutation
MODEL_ID=Qwen/Qwen2.5-7B-Instruct bash scripts/start_vllm.sh
```

### Step 4: Confirm The New Model Is Running

In another tmux pane:

```bash
curl http://localhost:8090/v1/models
```

The response should show the new model id.

### Step 5: Update `.env`

Open `.env`:

```bash
cd ~/ai-mutation
nano .env
```

Update:

```dotenv
AI_MODEL=<new-model-id>
```

Example:

```dotenv
AI_MODEL=Qwen/Qwen2.5-7B-Instruct
```

For a vision model, also update:

```dotenv
AI_VISION_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
```

Keep these unchanged:

```dotenv
AI_BASE_URL=http://localhost:8090/v1
BACKEND_URL=http://localhost:8000
```

### Step 6: Restart The Gradio UI

Move to the UI pane.

Stop it:

```text
Ctrl+C
```

Start it again:

```bash
cd ~/ai-mutation
bash scripts/start_ui.sh
```

FastAPI usually does not need to be restarted just because the model changed.
Restart FastAPI only if you changed backend code or environment variables that
the backend must reload.

### Step 7: Select The Model In The UI

Open:

```text
http://<droplet-public-ip>:8501
```

In the sidebar:

1. Go to the **Model** dropdown.
2. Select the model you started with vLLM.
3. If it is not shown, choose **Custom...** and type the exact model id.

The UI tries to read live models from:

```text
http://localhost:8090/v1/models
```

If vLLM is running correctly, the served model should appear in the dropdown.

### Step 8: Test With A Simple Mutation

Use:

```text
BRAF V600E
```

Click:

```text
Analyze
```

If it works, the new model is ready.

### Important Notes

One vLLM server usually serves one main model at a time. If you want a different
model, restart vLLM with that model.

The model id in `.env` is the default model shown by the app. If you select a
different model in the UI, that selected model is sent to FastAPI for that
analysis request. Either way, the selected model must be served by vLLM.

Changing the UI dropdown does not download or start a new model. It only tells
the app which model id to send to the already-running endpoint.

## 21. Snapshot And Resume The GPU Droplet


Yes. After your setup is working, you should create a **snapshot** of the GPU droplet. That lets you restore the droplet later without repeating all installation steps.

**Important idea:** a snapshot saves the droplet disk state, but it usually does **not** save currently running processes. So after restoring a snapshot, you normally restart:

```text
1. vLLM model server
2. FastAPI backend
3. Gradio UI
```

**Before Taking Snapshot**
SSH into the droplet:

```bash
ssh <user>@<droplet-public-ip>
```

Stop the running app services cleanly.

If you used `tmux`, reattach:

```bash
tmux attach -t ai-mutation
```

Then stop each running service with:

```text
Ctrl+C
```

Stop:

```text
vLLM
FastAPI
Gradio
```

Then exit tmux:

```bash
exit
```

Optional but recommended: confirm no app processes are running:

```bash
ps aux | grep -E "vllm|uvicorn|gradio"
```

**Take Snapshot In AMD Cloud UI**
In AMD AI Developer Cloud:

```text
1. Go to GPU Droplets
2. Select your droplet
3. Stop/Shut down the droplet if required
4. Choose Snapshot / Create Snapshot
5. Give it a name
```

Example name:

```text
ai-mutation-amd-ready-v1
```

Use a name that tells you what is inside.

Good examples:

```text
ai-mutation-base-python-v1
ai-mutation-vllm-qwen-ready-v1
ai-mutation-full-app-ready-v1
```

**After Snapshot Is Created**
You can either:

```text
Option 1: Keep the droplet stopped
Option 2: Destroy the droplet and recreate later from snapshot
```

Destroying saves GPU runtime cost, but make sure the snapshot finished successfully first.

**Resume Later From Snapshot**
When you want to use the app again:

```text
1. Go to AMD AI Developer Cloud
2. Create GPU droplet from snapshot
3. Start the droplet
4. Copy the new public IP
5. SSH into it
```

```bash
ssh <user>@<new-droplet-public-ip>
```

Go to project folder:

```bash
cd ~/ai-mutation
```

Start all services:

```bash
bash scripts/start_all_tmux.sh
```

Open browser:

```text
http://<new-droplet-public-ip>:8501
```

**One More Important Thing**
If your new droplet has a different public IP, the browser URL changes.

But inside `.env`, this can stay the same:

```dotenv
AI_BASE_URL=http://localhost:8090/v1
BACKEND_URL=http://localhost:8000
```

Because everything is running inside the same cloud droplet.

**Best Snapshot Timing**
Take snapshots at these milestones:

```text
Snapshot 1: OS + Docker + project copied
Snapshot 2: Python environment installed
Snapshot 3: Full app tested successfully
```

The best one to keep is:

```text
ai-mutation-full-app-ready-v1
```

That snapshot should let you recreate the droplet and restart the app with only the three server commands.
