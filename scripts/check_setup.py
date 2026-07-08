"""
check_setup.py — confirms LazySlide and its stack are working.
Run with:  python scripts/check_setup.py
"""
import platform
import lazyslide as zs
import wsidata
import torch
import openslide

print("=" * 45)
print(" LazySlide setup check")
print("=" * 45)
print(f"Python           : {platform.python_version()}")
print(f"LazySlide        : {zs.__version__}")
print(f"wsidata          : {wsidata.__version__}")
print(f"PyTorch          : {torch.__version__}")
print(f"OpenSlide        : {openslide.__library_version__}")
gpu = "AVAILABLE (Apple MPS)" if torch.backends.mps.is_available() else "CPU only"
print(f"GPU acceleration : {gpu}")
print("=" * 45)
print("All good — you're ready to work on slides in ~/thesis/slides")
