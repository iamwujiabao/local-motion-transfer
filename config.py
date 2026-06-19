"""Central configuration for the motion transfer app."""
from pathlib import Path

# --- Paths -----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
WORK_DIR = DATA_DIR / "work"
THIRD_PARTY_DIR = ROOT / "third_party"

for _d in (UPLOAD_DIR, OUTPUT_DIR, WORK_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Device / precision ----------------------------------------------------
DEVICE = "cuda"            # "cpu" only for plumbing tests (MimicMotion is GPU-bound)
FP16 = True                # half precision; recommended
DEFAULT_STEPS = 25         # MimicMotion num_inference_steps
DEFAULT_SEED = 42

# --- MimicMotion integration ----------------------------------------------
# Put the extracted repo here (i.e. third_party/MimicMotion/inference.py exists),
# and download its weights into its own models/ dir per MimicMotion's README.
MIMICMOTION_REPO = THIRD_PARTY_DIR / "MimicMotion"
MM_BASE_MODEL_PATH = "stabilityai/stable-video-diffusion-img2vid-xt-1-1"
MM_CKPT_PATH = MIMICMOTION_REPO / "models" / "MimicMotion_1-1.pth"
MM_DWPOSE_DIR = MIMICMOTION_REPO / "models" / "DWPose"  # yolox_l.onnx + dw-ll_ucoco_384.onnx

# MimicMotion inference params. Defaults tuned toward lower VRAM (~12GB).
MM_RESOLUTION = 576        # short side; MimicMotion derives the other side from 9:16
MM_SAMPLE_STRIDE = 2       # frame decimation of the driving video
MM_TILE_SIZE = 16          # frames per tile (num_frames). 16 = low VRAM; 72 = full quality (16GB+)
MM_FRAMES_OVERLAP = 4      # tile overlap; must be < MM_TILE_SIZE
MM_NOISE_AUG = 0.0
MM_GUIDANCE = 2.0
MM_OUTPUT_FPS = 15         # playback fps of the generated clip

# --- Uploads ---------------------------------------------------------------
ALLOWED_IMAGE_EXT = {".png", ".jpg", ".jpeg"}
ALLOWED_VIDEO_EXT = {".mp4", ".mov"}
MAX_VIDEO_SECONDS = 30

# Longest side used ONLY for the cheap human-presence prefilter (not the model path;
# MimicMotion does its own resize/crop from the original files).
MAX_SIDE = 512

# --- Labeling (do not disable for shared/published output) -----------------
WATERMARK_TEXT = "AI-GENERATED"
METADATA_NOTE = (
    "AI-generated synthetic video produced by a local motion-transfer tool. "
    "The depicted motion was transferred from a separate driving video."
)
