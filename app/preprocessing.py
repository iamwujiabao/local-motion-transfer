"""Video preprocessing utilities (resolution, audio, frames, validation).

These are general-purpose media helpers built on OpenCV and the ffmpeg binary
bundled with imageio-ffmpeg, so no system ffmpeg install is required.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np
import imageio_ffmpeg

import config

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


# --- Resolution ------------------------------------------------------------
def cap_resolution(frame: np.ndarray, max_side: int = config.MAX_SIDE) -> np.ndarray:
    """Downscale a frame so its longest side <= max_side. Preserves aspect."""
    h, w = frame.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return frame
    scale = max_side / longest
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    # keep dimensions even (many encoders/models require it)
    new_w -= new_w % 2
    new_h -= new_h % 2
    return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)


# --- Frame I/O -------------------------------------------------------------
def read_frames(video_path: Path, max_side: int = config.MAX_SIDE):
    """Yield BGR frames from a video, downscaled to max_side."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield cap_resolution(frame, max_side)
    finally:
        cap.release()


def get_fps(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.release()
    return fps if fps > 0 else 24.0


def get_duration(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return (frames / fps) if fps > 0 else 0.0


def write_video(frames, out_path: Path, fps: float) -> Path:
    """Write a list/iterable of BGR frames to a silent mp4."""
    frames = list(frames)
    if not frames:
        raise ValueError("No frames to write.")
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    return out_path


# --- Audio (ffmpeg) --------------------------------------------------------
def extract_audio(video_path: Path, out_audio: Path) -> Path | None:
    """Extract the audio track to .aac. Returns None if the video has no audio."""
    cmd = [
        FFMPEG, "-y", "-i", str(video_path),
        "-vn", "-acodec", "aac", str(out_audio),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not out_audio.exists() or out_audio.stat().st_size == 0:
        return None
    return out_audio


def mux_audio(video_path: Path, audio_path: Path, out_path: Path) -> Path:
    """Combine a silent video with an audio track (audio trimmed to video length)."""
    cmd = [
        FFMPEG, "-y", "-i", str(video_path), "-i", str(audio_path),
        "-c:v", "copy", "-c:a", "aac", "-shortest", str(out_path),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return out_path


# --- Validation ------------------------------------------------------------
_hog = cv2.HOGDescriptor()
_hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def has_person(video_path: Path, sample_frames: int = 12) -> bool:
    """Cheap sanity check: sample frames and look for a human silhouette.

    Uses OpenCV's built-in HOG people detector. It is approximate — meant to
    reject obviously human-free clips early, not to be authoritative.
    """
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total == 0:
        cap.release()
        return False
    step = max(1, total // sample_frames)
    hits = 0
    for i in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = cap.read()
        if not ok:
            continue
        frame = cap_resolution(frame, 512)
        rects, _ = _hog.detectMultiScale(frame, winStride=(8, 8))
        if len(rects) > 0:
            hits += 1
    cap.release()
    return hits > 0


def validate_inputs(image_path: Path, video_path: Path) -> None:
    """Raise ValueError with a user-friendly message if inputs are unusable."""
    if image_path.suffix.lower() not in config.ALLOWED_IMAGE_EXT:
        raise ValueError(f"Image must be one of {sorted(config.ALLOWED_IMAGE_EXT)}")
    if video_path.suffix.lower() not in config.ALLOWED_VIDEO_EXT:
        raise ValueError(f"Video must be one of {sorted(config.ALLOWED_VIDEO_EXT)}")

    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError("Could not read the source image.")

    dur = get_duration(video_path)
    if dur <= 0:
        raise ValueError("Could not read the driving video.")
    if dur > config.MAX_VIDEO_SECONDS:
        raise ValueError(
            f"Driving video is {dur:.0f}s; max is {config.MAX_VIDEO_SECONDS}s. "
            "Trim it and try again."
        )
    if not has_person(video_path):
        raise ValueError(
            "No clear human figure detected in the driving video. "
            "Pose transfer needs a visible person to track."
        )
