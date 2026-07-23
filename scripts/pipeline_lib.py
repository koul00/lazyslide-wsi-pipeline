"""
pipeline_lib.py 

Each stage is a standalone function that reads/writes the same `<name>.zarr` store
and (optionally) saves a preview PNG to the output directory. Nothing is trained.

Typical use (from Python or a notebook):

    import pipeline_lib as pl
    cfg = pl.Config(model="plip")
    pl.preprocess("slides/tumor_084.tif", cfg)
    pl.features("slides/tumor_084.tif", cfg)
    pl.zero_shot("slides/tumor_084.tif", cfg)
    auc = pl.validate("slides/tumor_084.tif", cfg)   # None if no annotation
    pl.spatial("slides/tumor_084.tif", cfg)

    # many slides:
    summary = pl.run_cohort("slides", cfg)           # -> pandas DataFrame
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import lazyslide as zs


# ============================================================================
#  Configuration
# ============================================================================
@dataclass
class Config:

    """All pipeline parameters."""
    tile_px: int = 256        # tile size in pixels
    tile_mpp: float = 0.5     # microns/pixel (~20x)
    model: str = "plip"       # vision(+language) model, e.g. "plip" / "conch"
    resolution: float = 0.2   # spatial-domain clustering resolution

    classes: dict[str, str] = field(default_factory=lambda: {
        "tumor":  "metastatic carcinoma, malignant tumor cells, atypical epithelial cells",
        "normal": "normal lymph node tissue, small mature lymphocytes",
    })
    validate_class: str = "tumor"

    # which annotation label(s) mark the positive (tumor) region (case-insensitive
    # prefix match); empty list = treat every annotated region as positive
    positive_labels: list[str] = field(default_factory=lambda: ["tumor", "metastas"])

    # mixed precision for feature extraction; None -> auto-detect (GPU on, CPU off)
    amp: bool | None = None

    def __post_init__(self):
        if self.amp is None:
            self.amp = _gpu_available()

    @property
    def labels(self) -> list[str]:
        return list(self.classes.keys())


def _gpu_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available()
                    or (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()))
    except Exception:
        return False


# ============================================================================
#  Small helpers
# ============================================================================
def store_path(slide, out_dir="outputs") -> Path:
    """Path to the .zarr store for a slide."""
    return Path(out_dir) / f"{Path(slide).stem}.zarr"


def open_slide(slide, out_dir="outputs"):
    """Open a WSI backed by its .zarr store so results persist on wsi.write()."""
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    return zs.open_wsi(str(slide), backed_file=str(store_path(slide, out_dir)))


def _save(fig_path: Path):
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()


# ============================================================================
#  Annotation loading (for validation) — .xml (ASAP/CAMELYON16) & .geojson (QuPath)
# ============================================================================
def positive_match(label, cfg: Config) -> bool:
    if not cfg.positive_labels:
        return True
    label = (label or "").lower()
    return any(label.startswith(p.lower()) for p in cfg.positive_labels)


def load_annotation(path, cfg: Config):
    """Return (unioned positive geometry, n_regions) in level-0 pixel coords."""
    from shapely.geometry import Polygon, shape
    from shapely.ops import unary_union

    path = Path(path)
    if path.suffix.lower() == ".xml":            # ASAP / CAMELYON16
        root = ET.parse(path).getroot()
        polys = []
        for a in root.findall(".//Annotation"):
            if positive_match(a.get("PartOfGroup"), cfg):
                pts = [(float(c.get("X")), float(c.get("Y"))) for c in a.findall(".//Coordinate")]
                if len(pts) >= 3:
                    polys.append(Polygon(pts))
    else:                                        # QuPath-style GeoJSON
        data = json.loads(path.read_text())
        feats = data.get("features", data if isinstance(data, list) else [data])
        polys = []
        for f in feats:
            props = f.get("properties", {}) or {}
            cls = (props.get("classification") or {}).get("name") or props.get("name")
            if not positive_match(cls, cfg):
                continue
            geom = shape(f["geometry"])
            polys.extend(getattr(geom, "geoms", [geom]))
    return unary_union([p.buffer(0) for p in polys]), len(polys)


def find_annotation(slide):
    """Same-basename annotation next to the slide (.xml/.geojson/.json), or None."""
    slide = Path(slide)
    for ext in (".xml", ".geojson", ".json"):
        cand = slide.with_suffix(ext)
        if cand.exists():
            return cand
    return None


# ============================================================================
#  Stage 1 — preprocess
# ============================================================================
def preprocess(slide, cfg: Config, out_dir="outputs", save_plots=True):
    slide = Path(slide)
    out = Path(out_dir); out.mkdir(exist_ok=True, parents=True)
    wsi = open_slide(slide, out_dir)

    mpp = wsi.properties.mpp
    mpp_str = f"{mpp:.3f}" if mpp is not None else "UNKNOWN"
    print(f"opened {slide.stem}: {wsi.properties.shape}  mpp={mpp_str}")
    if mpp is None:
        print("WARNING: no mpp metadata — tiling at a micron target may fail.")

    zs.pp.find_tissues(wsi)
    print(f"tissue pieces: {len(wsi.shapes['tissues'])}")
    zs.pp.tile_tissues(wsi, tile_px=cfg.tile_px, mpp=cfg.tile_mpp)
    print(f"tiles: {len(wsi.shapes['tiles'])}")

    if save_plots:
        zs.pl.tissue(wsi); _save(out / f"{slide.stem}_tissue.png")
        zs.pl.tiles(wsi, linewidth=0.5); _save(out / f"{slide.stem}_tiles.png")

    wsi.write(overwrite=True)
    print(f"[preprocess] saved -> {store_path(slide, out_dir)}")
    return wsi


# ============================================================================
#  Stage 2 — features + spatial domains
# ============================================================================
def features(slide, cfg: Config, out_dir="outputs", save_plots=True, force=False):
    slide = Path(slide)
    out = Path(out_dir)
    wsi = open_slide(slide, out_dir)
    print(f"tiles: {len(wsi['tiles'])}")

    if force or f"{cfg.model}_tiles" not in wsi.tables:
        zs.tl.feature_extraction(wsi, cfg.model, amp=cfg.amp)   # the slow step
    else:
        print(f"reusing existing {cfg.model} features")

    zs.tl.feature_aggregation(wsi, feature_key=cfg.model)
    zs.pp.tile_graph(wsi)
    zs.tl.spatial_features(wsi, cfg.model)
    zs.tl.spatial_domain(wsi, feature_key=cfg.model, layer="spatial_features",
                         resolution=cfg.resolution)
    print(f"spatial domains: {wsi['tiles']['domain'].nunique()}")

    if save_plots:
        zs.pl.tiles(wsi, color="domain", alpha=0.5); _save(out / f"{slide.stem}_domains.png")

    wsi.write(overwrite=True)
    print(f"[features] {cfg.model} features + domains saved")
    return wsi


# ============================================================================
#  Stage 3 — zero-shot
# ============================================================================
def zero_shot(slide, cfg: Config, out_dir="outputs", save_plots=True):
    slide = Path(slide)
    out = Path(out_dir)
    wsi = open_slide(slide, out_dir)
    print(f"zero-shot model: {cfg.model}")

    if f"{cfg.model}_tiles" not in wsi.tables:
        zs.tl.feature_extraction(wsi, cfg.model, amp=cfg.amp)

    embeds = zs.tl.text_embedding([cfg.classes[k] for k in cfg.labels], model=cfg.model)
    embeds.index = cfg.labels
    zs.tl.text_image_similarity(wsi, embeds, model=cfg.model,
                                feature_key=f"{cfg.model}_tiles",
                                key_added="zeroshot", softmax=True)

    if save_plots:
        for lab in cfg.labels:
            zs.pl.tiles(wsi, feature_key="zeroshot", color=lab, style="heatmap")
            _save(out / f"{slide.stem}_zeroshot_{lab}.png")

    wsi.write(overwrite=True)
    print("[zero_shot] scores saved -> table 'zeroshot'")
    return wsi


# ============================================================================
#  Stage 4 — validate (optional; needs an annotation)
# ============================================================================
def validate(slide, cfg: Config, out_dir="outputs", annotation=None, save_plots=True):
    """Return ROC-AUC of the zero-shot score vs ground truth, or None if it can't
    be computed (no annotation, or only one class present)."""
    import geopandas as gpd
    from sklearn.metrics import roc_auc_score

    slide = Path(slide)
    out = Path(out_dir)
    ann = Path(annotation) if annotation else find_annotation(slide)
    if ann is None or not ann.exists():
        print(f"[validate] no annotation for {slide.stem} — skipping")
        return None

    wsi = open_slide(slide, out_dir)
    if "zeroshot" not in wsi.tables:
        print(f"[validate] no zero-shot scores for {slide.stem} — run zero_shot() first; skipping")
        return None
    tiles = wsi["tiles"].copy()
    zt = wsi.tables["zeroshot"]
    scores = zt.obs[["tile_id"]].copy()
    scores["score"] = np.asarray(zt[:, cfg.validate_class].X).ravel()
    tiles = tiles.merge(scores, on="tile_id", how="left")

    tumor, n = load_annotation(ann, cfg)
    print(f"loaded {n} positive region(s) from {ann.name}")

    centroids = gpd.GeoDataFrame(geometry=tiles.geometry.centroid, crs=tiles.crs)
    tiles["is_tumor"] = centroids.geometry.within(tumor).astype(int)

    ok = tiles["score"].notna()
    auc = None
    if tiles.loc[ok, "is_tumor"].nunique() < 2:
        print("only one class present — ROC-AUC undefined; showing heatmap only.")
    else:
        auc = roc_auc_score(tiles.loc[ok, "is_tumor"], tiles.loc[ok, "score"])
        mean_in = tiles.loc[ok & (tiles["is_tumor"] == 1), "score"].mean()
        mean_out = tiles.loc[ok & (tiles["is_tumor"] == 0), "score"].mean()
        print(f"mean '{cfg.validate_class}' INSIDE tumor : {mean_in:.4f}")
        print(f"mean '{cfg.validate_class}' OUTSIDE      : {mean_out:.4f}")
        print(f"ROC-AUC vs ground truth        : {auc:.3f}   (0.5=chance, 1.0=perfect)")

    if save_plots:
        zs.pl.tiles(wsi, feature_key="zeroshot", color=cfg.validate_class, style="heatmap")
        gpd.GeoSeries(tumor).boundary.plot(ax=plt.gca(), color="red", linewidth=1.2)
        title = f"zero-shot {cfg.validate_class} vs ground truth (red = tumor)"
        if auc is not None:
            title += f"\nAUC={auc:.3f}"
        plt.gca().set_title(title)
        _save(out / f"{slide.stem}_validation.png")
    return auc


# ============================================================================
#  Stage 5 — spatial (viz)
# ============================================================================
def spatial(slide, cfg: Config, out_dir="outputs", zoom_win=6000, save_plots=True):
    from matplotlib.collections import LineCollection

    slide = Path(slide)
    out = Path(out_dir)
    wsi = open_slide(slide, out_dir)
    tiles = wsi["tiles"]
    if "tiles_graph" not in wsi.tables or "domain" not in tiles.columns:
        raise RuntimeError("run features() first — it builds the tile graph and "
                           "'domain' labels this plot needs (they aren't persisted).")

    cent = tiles.geometry.centroid
    xy = np.column_stack([cent.x.to_numpy(), cent.y.to_numpy()])
    A = wsi.tables["tiles_graph"].obsp["spatial_connectivities"].tocoo()
    keep = A.row < A.col
    segments = np.stack([xy[A.row[keep]], xy[A.col[keep]]], axis=1)
    domain = tiles["domain"].astype(int).to_numpy()

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 10))
    ax0.add_collection(LineCollection(segments, colors="0.85", linewidths=0.2, zorder=1))
    sc = ax0.scatter(xy[:, 0], xy[:, 1], c=domain, cmap="tab10", s=5, zorder=2)
    ax0.set_title(f"{slide.stem}: {domain.max() + 1} spatial domains")
    plt.colorbar(sc, ax=ax0, label="spatial domain", shrink=0.5,
                 ticks=range(int(domain.max()) + 1))

    cx, cy = xy.mean(axis=0)
    win = zoom_win
    in_win = (np.abs(xy[:, 0] - cx) < win) & (np.abs(xy[:, 1] - cy) < win)
    edge_in = in_win[A.row[keep]] & in_win[A.col[keep]]
    ax1.add_collection(LineCollection(segments[edge_in], colors="0.5", linewidths=0.6, zorder=1))
    ax1.scatter(xy[in_win, 0], xy[in_win, 1], c=domain[in_win], cmap="tab10", s=45, zorder=2)
    ax1.set_xlim(cx - win, cx + win); ax1.set_ylim(cy - win, cy + win)
    ax1.set_title(f"zoom: neighbour graph ({int(edge_in.sum())} edges shown)")
    for ax in (ax0, ax1):
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")

    if save_plots:
        _save(out / f"{slide.stem}_spatial_graph.png")
    print(f"[spatial] nodes: {len(xy)}  edges: {len(segments)}  domains: {domain.max() + 1}")
    return wsi


# ============================================================================
#  a whole cohort
# ============================================================================
def run_slide(slide, cfg: Config, out_dir="outputs", force=False):
    """Full pipeline for ONE slide (preprocess -> features -> zero-shot -> validate).
    Returns a dict of results. Skips the heavy stages if scores already exist."""
    slide = Path(slide)
    wsi = open_slide(slide, out_dir)
    n_domains = None

    if force or "zeroshot" not in wsi.tables:
        preprocess(slide, cfg, out_dir, save_plots=True)
        # reuse the wsi that features() returns: its 'domain' column is still live
        # here (it may not survive a re-open), so record n_domains from it
        wsi = features(slide, cfg, out_dir, save_plots=True, force=force)
        if "domain" in wsi["tiles"].columns:
            n_domains = int(wsi["tiles"]["domain"].nunique())
        zero_shot(slide, cfg, out_dir, save_plots=True)

    auc = validate(slide, cfg, out_dir, save_plots=True)
    ann = find_annotation(slide)
    return {"slide": slide.stem, "n_tiles": len(wsi["tiles"]), "n_domains": n_domains,
            "annotation": ann.name if ann else None, "auc": auc}


def find_slides(slides_dir="slides"):
    """Every WSI in a directory (common extensions, case-insensitive), sorted."""
    exts = ("tif", "tiff", "svs", "ndpi")
    d = Path(slides_dir)
    found = set()
    for e in exts:
        found.update(d.glob(f"*.{e}"))
        found.update(d.glob(f"*.{e.upper()}"))
    return sorted(found)


def run_cohort(slides_dir, cfg: Config, out_dir="outputs", force=False):
    """Run the pipeline over every slide in a directory. Returns a pandas
    DataFrame (one row per slide) and writes outputs/cohort_summary.csv.
    One failing slide does not stop the rest."""
    import time
    import pandas as pd

    slides = find_slides(slides_dir)
    print(f"found {len(slides)} slide(s): {[p.name for p in slides]}")
    rows = []
    for i, s in enumerate(slides, 1):
        t0 = time.time()
        print(f"[{i}/{len(slides)}] {s.stem} ...", flush=True)
        try:
            r = run_slide(s, cfg, out_dir, force=force)
            r["seconds"] = round(time.time() - t0, 1)
            rows.append(r)
        except Exception as e:
            rows.append({"slide": s.stem, "auc": None, "error": repr(e)})
            print(f"   FAILED: {e}")

    summary = pd.DataFrame(rows)
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    summary.to_csv(Path(out_dir) / "cohort_summary.csv", index=False)
    print(f"saved -> {Path(out_dir) / 'cohort_summary.csv'}")
    return summary
