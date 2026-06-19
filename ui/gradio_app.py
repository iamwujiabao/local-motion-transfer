"""Gradio UI for the local motion transfer app.

Talks to the FastAPI backend over HTTP (start `uvicorn backend.main:app` first).
The consent checkbox is required before generation is allowed.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow running directly (python ui/gradio_app.py): put the project root on
# sys.path so top-level modules like `config` import regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr
import requests

import config

BACKEND = "http://127.0.0.1:8000"
POLL_SECONDS = 2
TIMEOUT_SECONDS = 60 * 30  # generation can be slow

CONSENT_LABEL = (
    "I confirm I have the right to use this likeness — it is my own image or "
    "a person who has consented — and I will not use the output to deceive or "
    "to depict anyone without their consent."
)


def _upload(kind: str, path: str) -> str:
    with open(path, "rb") as f:
        r = requests.post(
            f"{BACKEND}/upload",
            data={"kind": kind},
            files={"file": (Path(path).name, f)},
        )
    r.raise_for_status()
    return r.json()["file_id"]


def run(image_path, video_path, consent, steps, seed):
    """Generator: yields (status_markdown, output_video_path)."""
    if not consent:
        yield "⚠️ You must confirm the consent statement before generating.", None
        return
    if not image_path or not video_path:
        yield "⚠️ Please provide both a source image and a driving video.", None
        return

    try:
        yield "Uploading…", None
        image_id = _upload("image", image_path)
        video_id = _upload("video", video_path)

        yield "Queuing job…", None
        r = requests.post(
            f"{BACKEND}/generate",
            data={
                "image_id": image_id,
                "video_id": video_id,
                "consent_confirmed": "true",
                "steps": int(steps),
                "seed": int(seed),
            },
        )
        if r.status_code != 200:
            detail = r.json().get("detail", r.text)
            yield f"❌ {detail}", None
            return
        job_id = r.json()["job_id"]

        start = time.time()
        while True:
            if time.time() - start > TIMEOUT_SECONDS:
                yield "❌ Timed out waiting for generation.", None
                return
            s = requests.get(f"{BACKEND}/status/{job_id}").json()
            status = s["status"]
            if status == "error":
                yield f"❌ Generation failed: {s.get('error')}", None
                return
            if status == "done":
                break
            pos = s.get("queue_position", 0)
            ahead = f" ({pos} job(s) ahead)" if pos else ""
            yield f"⏳ {status}: {s.get('progress','')}{ahead}", None
            time.sleep(POLL_SECONDS)

        # download result
        out = config.OUTPUT_DIR / f"ui_{job_id}.mp4"
        with requests.get(f"{BACKEND}/download/{job_id}", stream=True) as dl:
            dl.raise_for_status()
            with open(out, "wb") as f:
                for chunk in dl.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
        yield "✅ Done. Output is labeled AI-GENERATED.", str(out)

    except requests.ConnectionError:
        yield (
            "❌ Cannot reach the backend. Start it with:\n"
            "`uvicorn backend.main:app --port 8000`",
            None,
        )
    except Exception as exc:  # noqa: BLE001
        yield f"❌ {exc}", None


with gr.Blocks(title="Local Motion Transfer") as demo:
    gr.Markdown(
        "# Local Motion Transfer\n"
        "Animate a source image with the motion from a driving video — all on "
        "your machine. Outputs are automatically labeled as AI-generated."
    )
    with gr.Row():
        with gr.Column():
            image_in = gr.Image(label="Source image (the person)", type="filepath")
            video_in = gr.Video(label="Driving video (the motion)")
            steps = gr.Slider(10, 50, value=config.DEFAULT_STEPS, step=1,
                              label="Inference steps (higher = slower, often cleaner)")
            seed = gr.Number(value=config.DEFAULT_SEED, label="Seed", precision=0)
            consent = gr.Checkbox(label=CONSENT_LABEL, value=False)
            go = gr.Button("Generate", variant="primary")
        with gr.Column():
            status = gr.Markdown("Idle.")
            video_out = gr.Video(label="Result (AI-generated)")

    go.click(
        run,
        inputs=[image_in, video_in, consent, steps, seed],
        outputs=[status, video_out],
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
