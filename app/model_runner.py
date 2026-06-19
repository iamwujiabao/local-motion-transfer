"""Model integration + pipeline orchestration, wired to MimicMotion.

The generation core is NOT reimplemented here. This module imports and calls
MimicMotion's own published functions (`preprocess`, `run_pipeline`,
`create_pipeline`) from the repo you place in `third_party/MimicMotion`. The
synthesis engine, its weights, and its license remain MimicMotion's; this file
is glue plus the app's labeling/audio handling.

Data flow (matches MimicMotion's inference.py):
    preprocess(video, image) -> (pose_pixels, image_pixels)   # pose is aligned
                                                              # to the ref body,
                                                              # so image+video are
                                                              # processed together
    run_pipeline(pipeline, image_pixels, pose_pixels, device, task) -> frames

We convert MimicMotion's output frames to BGR so the existing labeling, audio
re-muxing, and metadata-tagging steps apply automatically — i.e. every output
is still stamped AI-GENERATED and tagged in metadata.
"""
from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import List

import cv2
import numpy as np

import config
from app import preprocessing, watermark
from app.queue import Job

# Make the vendored repo importable (its modules use top-level imports like
# `from constants import ASPECT_RATIO` and run from the repo root).
if config.MIMICMOTION_REPO.exists():
    repo_str = str(config.MIMICMOTION_REPO)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

_PIPELINE = None  # cached across requests; the model is multi-GB


class ModelNotConfigured(RuntimeError):
    """Raised when the MimicMotion repo or its weights are not in place."""


def model_configured() -> tuple[bool, str]:
    """Check the repo and weights exist. Returns (ok, human-readable reason)."""
    repo = config.MIMICMOTION_REPO
    if not (repo / "inference.py").exists():
        return False, (
            f"MimicMotion repo not found at {repo}. Extract it so that "
            f"{repo / 'inference.py'} exists."
        )
    if not config.MM_CKPT_PATH.exists():
        return False, f"Missing MimicMotion checkpoint: {config.MM_CKPT_PATH}"
    for f in ("yolox_l.onnx", "dw-ll_ucoco_384.onnx"):
        if not (config.MM_DWPOSE_DIR / f).exists():
            return False, f"Missing DWPose weight: {config.MM_DWPOSE_DIR / f}"
    return True, "ok"


@contextmanager
def _in_repo_dir():
    """Run with cwd at the repo root so DWPose finds models/DWPose/*.onnx."""
    prev = os.getcwd()
    os.chdir(config.MIMICMOTION_REPO)
    try:
        yield
    finally:
        os.chdir(prev)


def _make_task(steps: int, seed: int):
    """Build the task-config object MimicMotion's run_pipeline expects."""
    from types import SimpleNamespace
    return SimpleNamespace(
        seed=seed,
        num_frames=config.MM_TILE_SIZE,
        frames_overlap=config.MM_FRAMES_OVERLAP,
        noise_aug_strength=config.MM_NOISE_AUG,
        num_inference_steps=steps,
        guidance_scale=config.MM_GUIDANCE,
    )


def _get_pipeline():
    """Lazily build and cache the MimicMotion pipeline (mirrors inference.main)."""
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    ok, reason = model_configured()
    if not ok:
        raise ModelNotConfigured(reason)

    try:
        import torch
        from omegaconf import OmegaConf
        # importing `inference` runs its top-level geglu patch (needed before
        # the pipeline is constructed) and exposes create_pipeline.
        from inference import create_pipeline
    except Exception as exc:  # noqa: BLE001
        raise ModelNotConfigured(
            "Could not import MimicMotion. Install its dependencies "
            "(decord, onnxruntime, omegaconf, diffusers, transformers, "
            "einops, accelerate, matplotlib) into this environment. See README. "
            f"Underlying error: {exc}"
        ) from exc

    if config.FP16:
        torch.set_default_dtype(torch.float16)

    infer_config = OmegaConf.create({
        "base_model_path": config.MM_BASE_MODEL_PATH,
        "ckpt_path": str(config.MM_CKPT_PATH),
    })
    with _in_repo_dir():
        _PIPELINE = create_pipeline(infer_config, config.DEVICE)
    return _PIPELINE


def _frames_to_bgr(frames_tensor) -> List[np.ndarray]:
    """MimicMotion returns uint8 RGB tensor (F, C, H, W) -> list of BGR HWC arrays."""
    out = []
    for f in frames_tensor:
        rgb = f.permute(1, 2, 0).cpu().numpy().astype(np.uint8)  # H, W, C (RGB)
        out.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return out


def generate_frames(image_path: Path, video_path: Path,
                    *, steps: int, seed: int, job: Job | None = None) -> List[np.ndarray]:
    """Run MimicMotion end-to-end and return generated BGR frames.

    Thin adapter over MimicMotion's own `preprocess` + `run_pipeline`.
    """
    import torch  # noqa: F401  (ensures torch import errors surface here)
    from inference import preprocess as mm_preprocess
    from inference import run_pipeline as mm_run_pipeline

    pipeline = _get_pipeline()
    task = _make_task(steps, seed)

    if job is not None:
        job.set_progress("Extracting pose (DWPose) + preprocessing")
    with _in_repo_dir(), torch.no_grad():
        # Note: original files are passed; MimicMotion does its own resize/crop.
        pose_pixels, image_pixels = mm_preprocess(
            str(video_path), str(image_path),
            resolution=config.MM_RESOLUTION, sample_stride=config.MM_SAMPLE_STRIDE,
        )
        if job is not None:
            job.set_progress("Generating frames (diffusion — the slow part)")
        frames_tensor = mm_run_pipeline(
            pipeline, image_pixels, pose_pixels, config.DEVICE, task
        )
    return _frames_to_bgr(frames_tensor)


def run_pipeline(job: Job, image_path: Path, video_path: Path,
                 *, steps: int = config.DEFAULT_STEPS,
                 seed: int = config.DEFAULT_SEED) -> Path:
    """Validate -> generate (MimicMotion) -> label -> re-attach audio -> tag.

    Returns the path to the finished, AI-labeled mp4. Signature unchanged from
    the stub, so the backend and UI need no edits.
    """
    job.set_progress("Validating inputs")
    preprocessing.validate_inputs(image_path, video_path)

    frames = generate_frames(image_path, video_path, steps=steps, seed=seed, job=job)
    if not frames:
        raise RuntimeError("Model returned no frames.")

    job.set_progress("Applying AI-generated label")
    frames = [watermark.stamp_frame(f) for f in frames]

    out_stem = config.OUTPUT_DIR / job.id
    silent = preprocessing.write_video(
        frames, out_stem.with_suffix(".silent.mp4"), fps=config.MM_OUTPUT_FPS
    )

    job.set_progress("Re-attaching audio")
    audio = preprocessing.extract_audio(video_path, config.WORK_DIR / f"{job.id}.aac")
    if audio is not None:
        source_for_tag = preprocessing.mux_audio(
            silent, audio, out_stem.with_suffix(".audio.mp4")
        )
    else:
        source_for_tag = silent

    job.set_progress("Embedding metadata")
    final = watermark.tag_metadata(source_for_tag, out_stem.with_suffix(".mp4"))

    for p in (silent, out_stem.with_suffix(".audio.mp4")):
        try:
            if p.exists() and p != final:
                p.unlink()
        except OSError:
            pass

    job.meta["output"] = str(final)
    job.set_progress("Done")
    return final
