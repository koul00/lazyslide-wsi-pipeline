"""
features.py — feature extraction + spatial-domain analysis.


"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lazyslide as zs

# --- settings you can change -------------------------------------------------
MODEL = "plip"       # vision(+language) model; also reused by zero_shot.py
RESOLUTION = 0.2     # spatial-domain clustering resolution (higher = more domains)
# -----------------------------------------------------------------------------

slide = Path(sys.argv[1] if len(sys.argv) > 1 else "slides/tumor_084.tif")
name = slide.stem
outputs = Path("outputs"); outputs.mkdir(exist_ok=True)
store = outputs / f"{name}.zarr"

wsi = zs.open_wsi(str(slide), backed_file=str(store))
print(f"opened {name}, tiles: {len(wsi['tiles'])}")

# 1. morphological features: one vector per tile  -> table "<model>_tiles"
zs.tl.feature_extraction(wsi, MODEL, amp=True)

# 2. aggregate tile features into a single slide-level representation
zs.tl.feature_aggregation(wsi, feature_key=MODEL)

# 3. build a spatial neighbour graph over the tiles  -> table "tiles_graph"
zs.pp.tile_graph(wsi)

# 4. smooth features across neighbours (blends position + morphology)
#    -> layer "spatial_features" on the "<model>_tiles" table
zs.tl.spatial_features(wsi, MODEL)

# 5. cluster tiles into spatial domains  -> column "domain" on wsi["tiles"]
zs.tl.spatial_domain(wsi, feature_key=MODEL, layer="spatial_features", resolution=RESOLUTION)
print(f"spatial domains: {wsi['tiles']['domain'].nunique()}")

# 6. visualize the domains
zs.pl.tiles(wsi, color="domain", alpha=0.5)
plt.savefig(outputs / f"{name}_domains.png", dpi=150, bbox_inches="tight")
plt.close()

wsi.write(overwrite=True)
print(f"features ({MODEL}) + domains saved -> {store}")
