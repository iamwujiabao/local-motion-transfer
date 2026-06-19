"""Quick GPU / environment sanity check. Run before anything else."""
import sys


def main() -> int:
    try:
        import torch
    except ImportError:
        print("PyTorch is not installed. Install it for your CUDA version:")
        print("  https://pytorch.org/get-started/locally/")
        return 1

    print(f"torch version : {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    print(f"CUDA available: {cuda_ok}")

    if not cuda_ok:
        print("\nNo CUDA GPU detected. Inference will be extremely slow on CPU.")
        print("Check your driver + the CUDA build of torch.")
        return 1

    idx = torch.cuda.current_device()
    name = torch.cuda.get_device_name(idx)
    total_gb = torch.cuda.get_device_properties(idx).total_memory / 1024**3
    print(f"GPU           : {name}")
    print(f"VRAM          : {total_gb:.1f} GB")

    if total_gb < 11.5:
        print("\nWarning: under ~12GB VRAM. Keep MAX_SIDE low (e.g. 512) and FP16 on.")
    print("\nEnvironment looks ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
