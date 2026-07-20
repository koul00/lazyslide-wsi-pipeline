"""
validate.py — compare zero-shot tumor scores against CAMELYON16 ground truth.

Labels each tile inside/outside the annotated tumor (ASAP XML, level-0 pixels)
and measures how well the zero-shot score separates them (ROC-AUC). Saves an
overlay of the annotation on the heatmap.

"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.ops import unary_union
from sklearn.metrics import roc_auc_score

import lazyslide as zs

CLASS = "tumor"   # which zero-shot class column to validate

slide = Path(sys.argv[1] if len(sys.argv) > 1 else "slides/tumor_084.tif")
name = slide.stem
store = Path("outputs") / f"{name}.zarr"
xml = slide.with_suffix(".xml")

# 1. load slide + zero-shot scores, align scores to tiles by tile_id
wsi = zs.open_wsi(str(slide), backed_file=str(store))
tiles = wsi["tiles"].copy()
zt = wsi.tables["zeroshot"]
scores = zt.obs[["tile_id"]].copy()
scores["score"] = np.asarray(zt[:, CLASS].X).ravel()
tiles = tiles.merge(scores, on="tile_id", how="left")

# 2. parse tumor polygons from the ASAP XML (level-0 pixel coords)
root = ET.parse(xml).getroot()
polys = []
for a in root.findall(".//Annotation"):
    if (a.get("PartOfGroup") or "").lower().startswith("tumor"):
        pts = [(float(c.get("X")), float(c.get("Y"))) for c in a.findall(".//Coordinate")]
        if len(pts) >= 3:
            polys.append(Polygon(pts))
tumor = unary_union([p.buffer(0) for p in polys])   # heal self-intersections
print(f"{len(polys)} tumor polygons loaded")

# 3. label each tile by whether its centroid falls inside the tumor annotation
centroids = gpd.GeoDataFrame(geometry=tiles.geometry.centroid, crs=tiles.crs)
tiles["is_tumor"] = centroids.geometry.within(tumor).astype(int)
print(f"tiles inside tumor: {tiles['is_tumor'].sum()}   outside: {(1 - tiles['is_tumor']).sum()}")

# 4. does the zero-shot score separate tumor vs non-tumor?
ok = tiles["score"].notna()
auc = roc_auc_score(tiles.loc[ok, "is_tumor"], tiles.loc[ok, "score"])
mean_in = tiles.loc[ok & (tiles["is_tumor"] == 1), "score"].mean()
mean_out = tiles.loc[ok & (tiles["is_tumor"] == 0), "score"].mean()
print("\n================  RESULT  ================")
print(f"  mean '{CLASS}' score INSIDE tumor : {mean_in:.4f}")
print(f"  mean '{CLASS}' score OUTSIDE      : {mean_out:.4f}")
print(f"  ROC-AUC vs ground truth          : {auc:.3f}")
print("  (0.5 = no signal, 1.0 = perfect)")
print("==========================================")

# 5. overlay the ground-truth tumor outline on the heatmap
zs.pl.tiles(wsi, feature_key="zeroshot", color=CLASS, style="heatmap")
gpd.GeoSeries(tumor).boundary.plot(ax=plt.gca(), color="red", linewidth=1.2)
plt.gca().set_title(f"zero-shot {CLASS} vs ground truth (red = tumor)\nAUC={auc:.3f}")
out = Path("outputs") / f"{name}_validation.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print("saved overlay ->", out)
