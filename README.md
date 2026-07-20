# Zero-shot tumor detection on whole-slide images with LazySlide

A [LazySlide](https://lazyslide.readthedocs.io) pipeline for WSIs. 

It takes a raw slide to tissue detection, tiling, foundation-model features extraction,  spatial-domain analysis and a training-free (zero-shot) tumor score that is validated
against the official (CAMELYON16 here) ground-truth annotation.



## Repository layout

```
notebooks/   pipeline.ipynb (primary entry point)
scripts/     the same pipeline split into per-stage scripts + a setup check
outputs/     result previews (.png) committed for reference on WSI -084
slides/      WSIs (.tif/.svs) + annotations (.xml/.geojson)  
```
----

## Setup

The tooling lives in a conda environment (Python 3.11, LazySlide 0.12,
PyTorch with Apple-MPS acceleration):

```bash
conda create -n lazyslide python=3.11
conda activate lazyslide
pip install "lazyslide[all,models]"   # core + foundation-model dependencies
pip install leidenalg igraph          # spatial-domain clustering (step 2)
python scripts/check_setup.py         # verifies LazySlide, PyTorch, OpenSlide, GPU
```

**Data.** Download slides into `slides/` and (optionally) their annotations
alongside, using the same basename as the slide. Two annotation formats are
supported for validation — `.xml` (ASAP / CAMELYON16) and `.geojson` (QuPath) —
so slides from other datasets can be validated too. 

CAMELYON16 ground-truth annotations are available at:

```
https://camelyon-dataset.s3.amazonaws.com/CAMELYON16/annotations/<name>.xml
```

## Pipeline


 1. Preprocess  open slide → detect tissue → cut into tiles  `outputs/<name>.zarr` (+ tissue/tiles previews) 
 2. Features + spatial PLIP features → aggregation → tile graph → spatial features → spatial domains table `plip_tiles`, `domain` per tile 
 3. Zero-shot score tiles against text prompts (no training)  table `zeroshot` (per-class heatmaps) 
 4. Validate  compare zero-shot score vs annotation (`.xml`/`.geojson`) → ROC-AUC  validation overlay + AUC (skipped if no annotation) 
 5. Spatial (viz)  draw the tile neighbour graph + spatial domains  neighbour-graph plot 

The same five stages also exist as standalone scripts in `scripts/` to
run from the terminal:

```bash
python scripts/preprocess.py slides/tumor_084.tif
python scripts/features.py   slides/tumor_084.tif
python scripts/zero_shot.py  slides/tumor_084.tif
python scripts/validate.py   slides/tumor_084.tif
python scripts/spatial.py    slides/tumor_084.tif   # optional
```

Reload a saved result:

```python
import lazyslide as zs
wsi = zs.open_wsi("slides/tumor_084.tif", backed_file="outputs/tumor_084.zarr")
# tissues, tiles (+ domain), plip_tiles, zeroshot are all inside
```

## Results (`tumor_084`)

- Slide: 40× (mpp 0.226), 86016 × 65536 px → **6125 tiles**.
- Zero-shot tumor score vs ground truth: **ROC-AUC 0.946**.
- Mean score inside tumor 0.516 vs outside 0.500 — the absolute gap is small, but
  the *ranking* of scores separates tumor from normal well.
- Predicted hotspots co-localize with the annotated macro-metastasis.



## Foundation models

- **Open (work out of the box):** `phikon` (vision), `plip` (vision+language).
- **Gated (require a HuggingFace account + access):** `conch`, `uni`, `virchow`,
  `gigapath`, `prism`, `titan`.

Zero-shot needs a **vision+language** model (`plip`, `conch`); vision-only models
provide tile features but cannot score against text.
