"""Quick GPU / system-RAM / environment sanity check. Run before anything else."""
import sys


def _system_ram_gb():
    """Best-effort total system RAM in GB (Linux/WSL via /proc/meminfo)."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) / 1024**2  # kB -> GB
    except OSError:
        pass
    try:
        import os
        return (os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")) / 1024**3
    except (ValueError, OSError, AttributeError):
        return None


def main() -> int:
    ok = True

    # --- System RAM ---------------------------------------------------------
    # The base model is deserialized into host RAM before it reaches the GPU,
    # so too little RAM => the process is OOM-killed (a bare "Killed", no stack).
    ram_gb = _system_ram_gb()
    if ram_gb is not None:
        print(f"System RAM    : {ram_gb:.1f} GB")
        if ram_gb < 15.0:
            ok = False
            print(
                "\nWarning: under ~16GB system RAM. Loading the SVD-XT base model\n"
                "(~10GB of fp16 weights) can be OOM-killed (a bare 'Killed' with no\n"
                "traceback; confirm via `dmesg | grep -i oom`). On WSL2, raise the VM\n"
                "limit in C:\\Users\\<you>\\.wslconfig:\n"
                "    [wsl2]\n    memory=12GB\n    swap=32GB\n"
                "then run `wsl --shutdown` and reopen. A large swap file also helps.\n"
            )
    else:
        print("System RAM    : (could not determine)")

    # --- PyTorch / CUDA -----------------------------------------------------
    try:
        import torch
    except ImportError:
        print("\nPyTorch is not installed. Run `uv sync` "
              "(or install torch for your CUDA version).")
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
        print("\nWarning: under ~12GB VRAM. Keep MM_TILE_SIZE / MM_RESOLUTION low "
              "(e.g. 16 / 576) and FP16 on (see config.py).")

    print("\nEnvironment looks ready." if ok
          else "\nEnvironment usable, but address the warning(s) above first.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
