"""
zero_shot.py — score every tile against text prompts (no training).
    plip and  CONCH
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import lazyslide as zs

# --- the classes to score each tile against (prompt wording matters a lot) ---
CLASSES = {
    "tumor":  "metastatic carcinoma, malignant tumor cells, atypical epithelial cells",
    "normal": "normal lymph node tissue, small mature lymphocytes",
}
# -----------------------------------------------------------------------------

slide = Path(sys.argv[1] if len(sys.argv) > 1 else "slides/tumor_084.tif")
MODEL = sys.argv[2] if len(sys.argv) > 2 else "plip"   # e.g. "plip" or "conch"
name = slide.stem
outputs = Path("outputs"); outputs.mkdir(exist_ok=True)
store = outputs / f"{name}.zarr"

wsi = zs.open_wsi(str(slide), backed_file=str(store))
labels = list(CLASSES.keys())
print(f"zero-shot model: {MODEL}")

# 1. reuse the model's tile features if step 02 made them, else extract now
if f"{MODEL}_tiles" not in wsi.tables:
    zs.tl.feature_extraction(wsi, MODEL, amp=True)

# 2. embed the class prompts with the model's text encoder
embeds = zs.tl.text_embedding([CLASSES[k] for k in labels], model=MODEL)
embeds.index = labels

# 3. per-tile probability for each class -> table "zeroshot"
zs.tl.text_image_similarity(
    wsi, embeds, model=MODEL, feature_key=f"{MODEL}_tiles",
    key_added="zeroshot", softmax=True,
)

# 4. save a heatmap per class
for lab in labels:
    zs.pl.tiles(wsi, feature_key="zeroshot", color=lab, style="heatmap")
    plt.savefig(outputs / f"{name}_zeroshot_{lab}.png", dpi=150, bbox_inches="tight")
    plt.close()

wsi.write(overwrite=True)
print(f"zero-shot scores saved -> table 'zeroshot' in {store}")
