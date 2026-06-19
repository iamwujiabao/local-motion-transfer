"""AI-generated labeling for synthetic outputs.

Two layers:
  1. A visible on-frame mark (cv2.putText — font-free, always works).
  2. An embedded metadata note via ffmpeg.

This is a transparency measure, not an anti-tamper control: re-encoding can
remove both. Keep it enabled for anything you share or publish.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import cv2
import numpy as np
import imageio_ffmpeg

import config

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def stamp_frame(frame: np.ndarray, text: str = config.WATERMARK_TEXT) -> np.ndarray:
    """Draw a small semi-transparent label in the bottom-right corner."""
    out = frame.copy()
    h, w = out.shape[:2]
    scale = max(0.4, w / 1280)
    thickness = max(1, int(round(scale * 2)))
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)

    pad = int(8 * scale)
    x2, y2 = w - pad, h - pad
    x1, y1 = x2 - tw - 2 * pad, y2 - th - base - 2 * pad

    overlay = out.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, out, 0.55, 0, out)
    cv2.putText(
        out, text, (x1 + pad, y2 - pad - base),
        cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA,
    )
    return out


def tag_metadata(in_path: Path, out_path: Path,
                 note: str = config.METADATA_NOTE) -> Path:
    """Stream-copy the video while embedding a metadata comment + tags."""
    cmd = [
        FFMPEG, "-y", "-i", str(in_path),
        "-c", "copy",
        "-metadata", f"comment={note}",
        "-metadata", "generated_by=local-motion-transfer",
        "-metadata", "synthetic=true",
        str(out_path),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return out_path
