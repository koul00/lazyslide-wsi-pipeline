# Zero-shot tumor detection on whole-slide images with LazySlide

A clean, reproducible [LazySlide](https://lazyslide.readthedocs.io) pipeline for
computational pathology on CAMELYON16 whole-slide images (WSIs). It takes a raw
slide to tissue detection, tiling, foundation-model features, spatial-domain
analysis, and a **training-free (zero-shot)** tumor score that is validated
against the official CAMELYON16 ground-truth annotation.

**Status:** proof of concept on one slide (`tumor_084`). The zero-shot detector
reaches **ROC-AUC 0.946** against the expert annotation, with no training.

---

## Repository layout

```
scripts/     numbered pipeline (01–05) + a setup check
outputs/     result previews (.png) committed for reference
slides/      WSIs (.tif/.svs) + annotations (.xml)  — NOT committed (large)
notebooks/   scratch notebooks (unused)
```

Whole-slide images and the derived `.zarr` feature store are **not** included in
this repository (a single WSI is ~800 MB); see *Setup* for how to obtain them.

---

## Setup

The tooling lives in a dedicated conda environment (Python 3.11, LazySlide 0.12,
PyTorch with Apple-MPS acceleration):

```bash
conda create -n lazyslide python=3.11
conda activate lazyslide
pip install "lazyslide[all,models]"   # core + foundation-model dependencies
pip install leidenalg igraph          # spatial-domain clustering (step 2)
python scripts/check_setup.py         # verifies LazySlide, PyTorch, OpenSlide, GPU
```

**Data.** Download CAMELYON16 slides into `slides/` and their annotations
alongside (same basename, `.xml`). Ground-truth annotations are available at:

```
https://camelyon-dataset.s3.amazonaws.com/CAMELYON16/annotations/<name>.xml
```

---

## Pipeline

Five small scripts in `scripts/`, each closely following the official docs.
Run them in order; each reads and writes the same `outputs/<name>.zarr` store.

| Step | Script | What it does | Key output |
|------|--------|--------------|------------|
| 1. Preprocess | `01_preprocess.py` | open slide → detect tissue → cut into tiles | `outputs/<name>.zarr` (+ tissue/tiles previews) |
| 2. Features + spatial | `02_features.py` | PLIP features → aggregation → tile graph → spatial features → spatial domains | table `plip_tiles`, `domain` per tile, `<name>_domains.png` |
| 3. Zero-shot | `03_zero_shot.py` | score tiles against text prompts (no training) | table `zeroshot`, `<name>_zeroshot_*.png` |
| 4. Validate | `04_validate.py` | compare zero-shot score vs CAMELYON16 annotation → ROC-AUC | `<name>_validation.png` |
| 5. Spatial (viz) | `05_spatial.py` | draw the tile neighbour graph + spatial domains | `<name>_spatial_graph.png` |

```bash
python scripts/01_preprocess.py slides/tumor_084.tif
python scripts/02_features.py   slides/tumor_084.tif
python scripts/03_zero_shot.py  slides/tumor_084.tif
python scripts/04_validate.py   slides/tumor_084.tif
python scripts/05_spatial.py    slides/tumor_084.tif   # optional
```

Reload a saved result:

```python
import lazyslide as zs
wsi = zs.open_wsi("slides/tumor_084.tif", backed_file="outputs/tumor_084.zarr")
# tissues, tiles (+ domain), plip_tiles, zeroshot are all inside
```

---

## Results (`tumor_084`)

- Slide: 40× (mpp 0.226), 86016 × 65536 px → **6125 tiles**.
- Zero-shot tumor score vs ground truth: **ROC-AUC 0.946**.
- Mean score inside tumor 0.516 vs outside 0.500 — the absolute gap is small, but
  the *ranking* of scores separates tumor from normal well.
- Predicted hotspots co-localize with the annotated macro-metastasis.

**Takeaway:** for this open vision–language model the *ranking* of tile scores,
not their absolute value, carries the tumor signal.

---

## Limitations

- **One slide** so far — a proof of concept, not evidence.
- An **easy case** (large macro-metastasis). Micrometastases and isolated tumor
  cells are substantially harder and may be missed.
- **Class imbalance** (99 tumor vs 6026 normal tiles) can inflate single-slide AUC.

---

## Foundation models

- **Open (work out of the box):** `phikon` (vision), `plip` (vision+language).
- **Gated (require a HuggingFace account + access):** `conch`, `uni`, `virchow`,
  `gigapath`, `prism`, `titan`.

Zero-shot needs a **vision+language** model (`plip`, `conch`); vision-only models
provide tile features but cannot score against text.

---

## Roadmap

1. **Foundation-model comparison** — evaluate stronger models (e.g. CONCH) for the
   zero-shot step; prompt phrasing strongly affects zero-shot quality.
2. **Cohort validation** — run the pipeline across many CAMELYON16 slides
   (including micrometastases and tumor-free nodes) and report AUC across the cohort.
3. **Trained model** — an attention-MIL head on the tile features for slide-level
   prediction.
