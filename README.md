# Zero-shot tumor detection on whole-slide images with LazySlide

A [LazySlide](https://lazyslide.readthedocs.io) pipeline for WSIs. 

It takes a raw slide to tissue detection, tiling, foundation-model features extraction,  spatial-domain analysis and a training-free (zero-shot) tumor score that is validated
against the official (CAMELYON16 here) ground-truth annotation.



## Repository layout

```
notebooks/   pipeline.ipynb        — imports the pipeline from scripts/ (primary entry point)
             pipeline_colab.ipynb  — self-contained version for Google Colab
scripts/     pipeline_lib.py       — the pipeline as importable functions (the notebook imports these)
             download_data.sh      — fetch CAMELYON16 slides + annotations
outputs/     result previews (.png) committed for reference on WSI -084
slides/      WSIs (.tif/.svs) + annotations (.xml/.geojson)
requirements.txt   pip dependencies
```
----

## Setup

The tooling lives in a conda environment (Python 3.11, LazySlide 0.12,
PyTorch with Apple-MPS acceleration):

```bash
conda create -n lazyslide python=3.11
conda activate lazyslide
pip install -r requirements.txt       # LazySlide + models + analysis deps
```

Reading `.tif`/`.svs` slides also needs the OpenSlide **system** library
(`brew install openslide` on macOS, `apt-get install openslide-tools` on Ubuntu/Colab).

**Data.** Fetch a CAMELYON16 slide + its annotation straight into `slides/`:

```bash
bash scripts/download_data.sh tumor_084      # -> slides/tumor_084.tif (+ tumor_084.xml)
```

Or add your own slides to `slides/` with an optional same-basename annotation.
Two formats are supported for validation — `.xml` (ASAP / CAMELYON16) and
`.geojson` (QuPath) — so slides from other datasets can be validated too.
CAMELYON16 annotations are also available directly at:

```
https://camelyon-dataset.s3.amazonaws.com/CAMELYON16/annotations/<name>.xml
```

## Pipeline

Run it from the notebook [`notebooks/pipeline.ipynb`](notebooks/pipeline.ipynb)
(primary entry point), or [`pipeline_colab.ipynb`](notebooks/pipeline_colab.ipynb) on Google Colab. Both drive the same five stages, all defined once in
[`scripts/pipeline_lib.py`](scripts/pipeline_lib.py):

 1. Preprocess  open slide → detect tissue → cut into tiles  `outputs/<name>.zarr` (+ tissue/tiles previews) 
 2. Features + spatial PLIP features → aggregation → tile graph → spatial features → spatial domains table `plip_tiles`, `domain` per tile 
 3. Zero-shot score tiles against text prompts (no training)  table `zeroshot` (per-class heatmaps) 
 4. Validate  compare zero-shot score vs annotation (`.xml`/`.geojson`) → ROC-AUC  validation overlay + AUC (skipped if no annotation) 
 5. Spatial (viz)  draw the tile neighbour graph + spatial domains  neighbour-graph plot 

To run a whole folder of slides at once (a cohort)
calls `pipeline_lib.run_cohort("slides", cfg)`, which runs every slide in `slides/`
and writes `outputs/cohort_summary.csv`.

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
