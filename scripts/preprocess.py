"""
preprocess.py — open a slide, find tissue, cut into tiles, save.

"""

import sys
from pathlib import Path

import lazyslide as zs
import matplotlib.pyplot as plt

# --- settings you can change -------------------------------------------------
TILE_PX = 256    # tile size in pixels
TILE_MPP = 0.5   # microns/pixel (~20x); the size most tile models expect
# -----------------------------------------------------------------------------

slide = Path(sys.argv[1] if len(sys.argv) > 1 else "slides/tumor_084.tif")
name = slide.stem
outputs = Path("outputs")
outputs.mkdir(exist_ok=True)
store = outputs / f"{name}.zarr"

# 1. open the slide, backed by a .zarr store so results persist on wsi.write()
wsi = zs.open_wsi(str(slide), backed_file=str(store))
print(f"opened {name}: {wsi.properties.shape}  mpp={wsi.properties.mpp:.3f}")

# 2. detect tissue  -> wsi.shapes["tissues"]
zs.pp.find_tissues(wsi)
print(f"tissue pieces: {len(wsi.shapes['tissues'])}")

# 3. tile the tissue -> wsi.shapes["tiles"]
zs.pp.tile_tissues(wsi, tile_px=TILE_PX, mpp=TILE_MPP)
print(f"tiles: {len(wsi.shapes['tiles'])}")

# 4. save preview images
zs.pl.tissue(wsi)
plt.savefig(outputs / f"{name}_tissue.png", dpi=150, bbox_inches="tight")
plt.close()

zs.pl.tiles(wsi, linewidth=0.5)
plt.savefig(outputs / f"{name}_tiles.png", dpi=150, bbox_inches="tight")
plt.close()

# 5. save the result to the .zarr store
wsi.write(overwrite=True)
print(f"saved -> {store}")
