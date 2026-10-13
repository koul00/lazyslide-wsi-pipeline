# Zero-shot tumor detection on whole-slide images with LazySlide

A [LazySlide](https://lazyslide.readthedocs.io) pipeline for WSIs.

It takes a raw slide to tissue detection, tiling, foundation-model feature
extraction, spatial-domain analysis and a training-free (zero-shot) tumor
score — validated against the official CAMELYON16 ground-truth annotation.
Validated both on a single slide (`tumor_084`) and on a 27-slide stratified
sample of the CAMELYON16 training set (see [Results](#results)).

## Repository layout

```
notebooks/   pipeline.ipynb        — imports the pipeline from scripts/ (primary entry point)
             pipeline_colab.ipynb  — self-contained version for Google Colab
scripts/     pipeline_lib.py       — the pipeline as importable functions (the notebook imports these)
             download_data.sh      — fetch CAMELYON16 slides + annotations
outputs/     result previews (.png) committed for reference on WSI tumor_084;
             everything else (.zarr stores, new previews, cohort_summary.csv)
             is regenerable and gitignored
slides/      WSIs (.tif/.svs) + annotations (.xml/.geojson) — gitignored (large)
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

**Data.** Fetch one or more CAMELYON16 slides + annotations straight into `slides/`:

```bash
bash scripts/download_data.sh tumor_084                       # one slide
bash scripts/download_data.sh tumor_001..010 normal_001..010  # a range, for a cohort
```

Or add your own slides to `slides/` with an optional same-basename annotation.
Two formats are supported for validation — `.xml` (ASAP / CAMELYON16) and
`.geojson` (QuPath) — so slides from other datasets can be validated too.
CAMELYON16 slides + annotations are also available directly at:

```
https://camelyon-dataset.s3.amazonaws.com/CAMELYON16/images/<name>.tif
https://camelyon-dataset.s3.amazonaws.com/CAMELYON16/annotations/<name>.xml
```
(mirrored on [GigaDB](https://gigadb.org/dataset/100439) via FTP, same files).

## Pipeline

Run it from the notebook [`notebooks/pipeline.ipynb`](notebooks/pipeline.ipynb)
(primary entry point), or [`pipeline_colab.ipynb`](notebooks/pipeline_colab.ipynb) on Google Colab. Both drive the same five stages, all defined once in
[`scripts/pipeline_lib.py`](scripts/pipeline_lib.py):

 1. Preprocess — open slide → detect tissue → cut into tiles → `outputs/<name>.zarr` (+ tissue/tiles previews)
 2. Features + spatial — PLIP features → aggregation → tile graph → spatial features → spatial domains → table `plip_tiles`, `domain` per tile
 3. Zero-shot — score tiles against text prompts (no training) → table `zeroshot` (per-class heatmaps)
 4. Validate — compare zero-shot score vs annotation (`.xml`/`.geojson`) → ROC-AUC → validation overlay + AUC (skipped if no annotation)
 5. Spatial (viz) — draw the tile neighbour graph + spatial domains → neighbour-graph plot

To run a whole folder of slides at once (a cohort), call
`pipeline_lib.run_cohort("slides", cfg)`, which runs every slide in `slides/`
(already-processed ones are cheap to re-touch, not redone) and writes
`outputs/cohort_summary.csv`.

Reload a saved result:

```python
import lazyslide as zs
wsi = zs.open_wsi("slides/tumor_084.tif", backed_file="outputs/tumor_084.zarr")
# tissues, tiles (+ domain), plip_tiles, zeroshot are all inside
```

## Results

### Single-slide deep dive (`tumor_084`)

- Slide: 40× (mpp 0.226), 86016 × 65536 px → **6125 tiles**.
- Zero-shot tumor score vs ground truth: **ROC-AUC 0.946**.
- Mean score inside tumor 0.516 vs outside 0.500 — the absolute gap is small, but
  the *ranking* of scores separates tumor from normal well.
- Predicted hotspots co-localize with the annotated macro-metastasis.

### Cohort validation (27-slide stratified sample)

A 10–20% stratified sample of the CAMELYON16 **training** set — 16 normal +
11 tumor slides, same tumor:normal ratio as the full set (111 tumor / 159
normal) — run through the same zero-shot pipeline, independently of
`tumor_084`.

| Slide | AUC |
|---|---|
| tumor_016 | 0.959 |
| tumor_058 | 0.952 |
| tumor_075 | 0.917 |
| tumor_038 | 0.931 |
| tumor_031 | 0.913 |
| tumor_005 | 0.898 |
| tumor_023 | 0.894 |
| tumor_093 | 0.881 |
| tumor_057 | 0.829 |
| tumor_065 | 0.804 |
| tumor_081 | 0.681 |

**Mean AUC 0.878, median 0.898** (range 0.681–0.959). Normal slides have no
AUC — there is no lesion to localize, so validation is skipped for them by
design (zero-shot scores are still computed).

Takeaway: the single-slide result was not a fluke — the zero-shot score
ranks tumor above normal tissue consistently across independent slides from
the same training set, with one clear outlier (`tumor_081`, 0.681) worth a
closer look (small/diffuse metastasis, staining variation, or an annotation
edge case).

## Foundation models

- **Open (work out of the box):** `phikon` (vision), `plip` (vision+language).
- **Available (HF access granted):** `uni` (vision) — as of 2026-10-13.
- **Still gated (require HF account + access):** `conch`, `virchow`,
  `gigapath`, `prism`, `titan`.

Zero-shot needs a **vision+language** model (`plip`, `conch`); vision-only
models (`phikon`, `uni`) provide tile features for aggregation/classification
but cannot score against text directly.
