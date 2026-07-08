"""
05_spatial.py — visualize the tile spatial graph and spatial domains.

Uses the outputs of step 02:
  - zs.pp.tile_graph      -> neighbour graph in tables["tiles_graph"].obsp
  - zs.tl.spatial_domain  -> "domain" label on wsi["tiles"]

Draws every tile as a node at its centroid, connects neighbouring tiles with
edges, and colours the nodes by spatial domain.

Usage (from ~/thesis, with the `lazyslide` conda env active):
    python scripts/05_spatial.py slides/tumor_084.tif
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import lazyslide as zs

slide = Path(sys.argv[1] if len(sys.argv) > 1 else "slides/tumor_084.tif")
name = slide.stem
outputs = Path("outputs"); outputs.mkdir(exist_ok=True)
store = outputs / f"{name}.zarr"

wsi = zs.open_wsi(str(slide), backed_file=str(store))
tiles = wsi["tiles"]

# node positions = tile centroids (level-0 pixel coordinates)
cent = tiles.geometry.centroid
xy = np.column_stack([cent.x.to_numpy(), cent.y.to_numpy()])

# edges = the neighbour graph; upper triangle so each edge is drawn once
A = wsi.tables["tiles_graph"].obsp["spatial_connectivities"].tocoo()
keep = A.row < A.col
segments = np.stack([xy[A.row[keep]], xy[A.col[keep]]], axis=1)

# one colour per spatial domain
domain = tiles["domain"].astype(int).to_numpy()

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 10))

# --- left: whole slide, nodes coloured by domain (edges faint) ---------------
ax0.add_collection(LineCollection(segments, colors="0.85", linewidths=0.2, zorder=1))
sc = ax0.scatter(xy[:, 0], xy[:, 1], c=domain, cmap="tab10", s=5, zorder=2)
ax0.set_title(f"{name}: {domain.max() + 1} spatial domains")
plt.colorbar(sc, ax=ax0, label="spatial domain", shrink=0.5,
             ticks=range(int(domain.max()) + 1))

# --- right: a zoom window where the neighbour edges are actually visible ------
cx, cy = xy.mean(axis=0)
win = 6000  # level-0 pixels (~10 tiles across) so individual edges show
in_win = (np.abs(xy[:, 0] - cx) < win) & (np.abs(xy[:, 1] - cy) < win)
edge_in = in_win[A.row[keep]] & in_win[A.col[keep]]
ax1.add_collection(LineCollection(segments[edge_in], colors="0.5", linewidths=0.6, zorder=1))
ax1.scatter(xy[in_win, 0], xy[in_win, 1], c=domain[in_win], cmap="tab10", s=45, zorder=2)
ax1.set_xlim(cx - win, cx + win); ax1.set_ylim(cy - win, cy + win)
ax1.set_title(f"zoom: neighbour graph ({int(edge_in.sum())} edges shown)")

for ax in (ax0, ax1):
    ax.set_aspect("equal")
    ax.invert_yaxis()                   # image coords: y increases downward
    ax.axis("off")

out = outputs / f"{name}_spatial_graph.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"nodes: {len(xy)}   edges: {len(segments)}   domains: {domain.max() + 1}")
print(f"saved -> {out}")
