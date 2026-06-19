# Local AI Motion Transfer (wired to MimicMotion)

A privacy-first, local-first app that animates a source image using the motion
from a driving video. The synthesis engine is **MimicMotion** (Tencent, ICML
2025), called in-process. All processing runs on your machine.

## Responsible use (read this first)

This tool synthesizes a video of a person performing movements they did not
perform. That is useful for animation, research, accessibility, and personal
creative work — and harmful when used to fabricate convincing video of real
people without their consent.

Two defaults are enforced and stay in place:

1. **Consent gate** — generation requires affirming you have the right to use
   the source likeness (your own image, or a consenting subject).
2. **AI-generated labeling** — every output is stamped "AI-GENERATED" on-frame
   and tagged in file metadata. The pipeline labels whatever frames the model
   returns, so it can't be skipped through normal use. (Re-encoding can strip
   it; this is transparency, not an anti-abuse control.)

Do not use this to depict identifiable real people without consent, to produce
sexual content involving anyone, or to deceive.

## How it's wired

`app/model_runner.py` imports and calls MimicMotion's own functions
(`preprocess`, `run_pipeline`, `create_pipeline`) — the generation logic is not
reimplemented. MimicMotion aligns the driving-video pose to the reference body,
so the reference image and the video are preprocessed together; the public
`run_pipeline(job, image, video)` entrypoint is unchanged, so the FastAPI
backend and its endpoints need no edits. Output frames are converted to BGR so
the labeling / audio / metadata steps apply automatically.

The UI is a single static HTML page (`frontend/index.html`) served by the
FastAPI app at `/` and talking to the same endpoints over `fetch()` — no
separate UI server, no front-end build step, no extra dependencies.

## Setup (uv)

MimicMotion's upstream README uses conda. You don't need it — this project
declares the whole stack (the app's web deps **and** MimicMotion's pinned
runtime) in `pyproject.toml`, so a single `uv sync` builds one environment that
runs the app and imports MimicMotion in-process.

> uv users: `pyproject.toml` is the source of truth. `requirements.txt` exists
> only for the legacy pip-without-uv path and is not needed here.

### 1. Vendor the MimicMotion repo

Place it so that `third_party/MimicMotion/inference.py` exists:

```bash
unzip MimicMotion-main.zip
mv MimicMotion-main third_party/MimicMotion
```

(Or point `MIMICMOTION_REPO` in `config.py` at wherever you keep it.)

### 2. Create the environment

```bash
uv python install 3.11     # fetch 3.11 if you don't have it (pinned in .python-version)
uv sync                    # creates .venv and installs everything from pyproject.toml
```

`uv sync` pulls `torch==2.0.1+cu117` / `torchvision==0.15.2+cu117` from the
PyTorch CUDA-11.7 index configured in `pyproject.toml`; all other packages come
from PyPI.

The `huggingface-hub` (<0.26) pin in `pyproject.toml` is load-bearing, not
cosmetic: `diffusers==0.27.0` imports `huggingface_hub.cached_download`, which
was removed in hub 0.26. Leave it in place unless you also bump diffusers.

### 3. Download weights (into the MimicMotion repo's own models/ dir)

Follow MimicMotion's README. The result must be:

```
third_party/MimicMotion/models/
├── DWPose/
│   ├── dw-ll_ucoco_384.onnx
│   └── yolox_l.onnx
└── MimicMotion_1-1.pth
```

The SVD base model (`stabilityai/stable-video-diffusion-img2vid-xt-1-1`) is
pulled from Hugging Face on first run.

### 4. Verify

```bash
uv run python scripts/check_env.py
```

## Run

```bash
uv run uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Then open **http://127.0.0.1:8000** — the backend serves both the API and the
UI, so it's a single process with no separate UI step. The page shows whether
the model is configured; if the repo or weights are missing, it (and
`/generate`) reports a precise "model not configured" message naming the missing
file.

### GPU / CUDA notes

- **GPU torch on a 40-series (Ada) card:** torch 2.0.1/cu117 predates mature Ada
  support and may warn or run slower. If you hit kernel errors, bump to a newer
  CUDA wheel by editing the `pytorch-cu117` index URL in `pyproject.toml`
  (e.g. `.../whl/cu121`) and the torch/torchvision pins — note this can pull
  diffusers/transformers compatibility along with it.
- **DWPose (onnxruntime):** runs on **CPU** by default. The `onnxruntime-gpu`
  wheels are hard-pinned to a CUDA major version (the PyPI default now links
  CUDA 12/13), which won't match this CUDA 11.7 stack and fails at import with a
  missing `libcudart.so`. DWPose runs only at preprocessing and is not the
  bottleneck, so CPU is the safe default; MimicMotion asks for the CUDA provider
  and onnxruntime falls back to CPU with a warning. For GPU pose (faster), use a
  CUDA-11 build such as `onnxruntime-gpu==1.16.3` and make CUDA 11 runtime libs
  visible to onnxruntime.

### System RAM (and WSL2)

The base model is deserialized into **system RAM** before it moves to the GPU, so
host memory matters as much as VRAM. SVD-XT is ~10GB of fp16 weights; with too
little RAM the load is **OOM-killed** — a bare `Killed` with no Python traceback
(confirm with `dmesg | grep -i oom`). Plan for **≥16GB** of RAM, or a large swap
file to absorb the load spike.

On **WSL2** the Linux VM is memory-capped (often a fraction of Windows RAM — check
with `free -h`). Raise it in `C:\Users\<you>\.wslconfig`:

```ini
[wsl2]
memory=12GB        # e.g. 24GB on a 32GB machine; leave Windows a few GB
swap=32GB          # swap absorbs the load spike and prevents the kill
```

Then, from Windows: `wsl --shutdown`, reopen the Ubuntu terminal, and confirm the
new ceiling with `free -h`. `scripts/check_env.py` reports system RAM too.

## VRAM and tuning (`config.py`)

MimicMotion 1.1 targets up to 72 frames at 576×1024; the VAE decode is the
memory bottleneck. Levers that take effect through this adapter:

- `MM_TILE_SIZE` — frames per tile (`num_frames`). Default **16** (lower VRAM);
  raise toward **72** on 16GB+ for best temporal quality.
- `MM_RESOLUTION` — short side (default 576). Lower it to cut memory.
- `MM_SAMPLE_STRIDE` — driving-video frame decimation.
- `MM_OUTPUT_FPS`, `MM_GUIDANCE`, `MM_NOISE_AUG`, `DEFAULT_STEPS`, `DEFAULT_SEED`.

Finer memory controls (`decode_chunk_size`, CPU VAE offload) live inside
MimicMotion's `inference.py`/pipeline and require editing that file; this
adapter calls `run_pipeline` as published (decode chunk fixed at 8).

## Known limitation: audio sync

MimicMotion decimates the driving video by `MM_SAMPLE_STRIDE`, so the generated
clip has fewer frames than the source. The original audio is re-attached and
trimmed to length (`-shortest`), but it is only **approximately** aligned to the
motion — there is no frame-accurate / lip-sync step, by design.

## Project layout

```
local-motion-transfer/
├── pyproject.toml          uv project: all deps + PyTorch cu117 index
├── .python-version         pins 3.11
├── .gitignore
├── requirements.txt        legacy pip path only (uv users ignore)
├── README.md
├── config.py               paths, device, MimicMotion params, labeling text
├── scripts/
│   └── check_env.py        GPU/CUDA + system-RAM check
├── app/
│   ├── __init__.py
│   ├── preprocessing.py     validation, human-presence prefilter, audio, frames
│   ├── model_runner.py      MimicMotion adapter + pipeline orchestration
│   ├── watermark.py         on-frame label + metadata tag
│   └── queue.py             single-GPU sequential job queue
├── backend/
│   ├── __init__.py
│   └── main.py              FastAPI: serves UI + /upload /generate /status /download
├── frontend/
│   └── index.html           static UI served at / (consent checkbox required)
├── third_party/
│   └── README.md            <- place MimicMotion repo + weights here
└── data/                    created at runtime (uploads / outputs / work)
```
