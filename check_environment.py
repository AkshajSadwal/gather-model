"""Check Python, PyTorch, and available accelerator (CUDA or Apple MPS)."""

import platform
import sys

print(f"Python: {sys.version.split()[0]}")
print(f"Platform: {platform.platform()}")

try:
    import torch
except ImportError:
    print("PyTorch: NOT INSTALLED")
    print("Install dependencies with: python -m pip install -r requirements.txt")
    raise SystemExit(1)

print(f"PyTorch: {torch.__version__}")
cuda_ok = torch.cuda.is_available()
mps_built = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_built())
mps_ok = bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
print(f"CUDA available: {cuda_ok}")
print(f"MPS built: {mps_built}")
print(f"MPS available: {mps_ok}")

if cuda_ok:
    print(f"Selected device: CUDA ({torch.cuda.get_device_name(0)})")
elif mps_ok:
    print("Selected device: Apple Silicon GPU (MPS)")
else:
    print("Selected device: CPU")
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        print("WARNING: Apple Silicon detected but MPS is unavailable. Do not start full training until this is fixed.")

for package in ("numpy", "pandas", "sklearn", "tqdm"):
    try:
        mod = __import__(package)
        print(f"{package}: {getattr(mod, '__version__', 'installed')}")
    except ImportError:
        print(f"{package}: NOT INSTALLED")
