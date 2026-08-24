import os
import pickle
import numpy as np
import rasterio
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt

DATA_DIR = os.path.expanduser("~/Desktop/FloodMappingSAR/data")
OUT_DIR = os.path.expanduser("~/Desktop/FloodMappingSAR/output")

pre_path = os.path.join(DATA_DIR, "hyderabad_preflood_2024-08-25.tif")
post_path = os.path.join(DATA_DIR, "hyderabad_postflood_2024-09-06.tif")
model_path = os.path.join(OUT_DIR, "flood_classifier_model_v2.pkl")

with open(model_path, "rb") as f:
    clf = pickle.load(f)
print(f"Loaded model trained on Hawke's Bay - applying to Hyderabad with NO retraining")

with rasterio.open(pre_path) as src:
    pre_vv = median_filter(src.read(1).astype(np.float32), size=5)
    pre_vh = median_filter(src.read(2).astype(np.float32), size=5)
    shape = src.shape
    profile = src.profile

with rasterio.open(post_path) as src:
    post_vv = median_filter(src.read(1).astype(np.float32), size=5)
    post_vh = median_filter(src.read(2).astype(np.float32), size=5)

vv_diff = post_vv - pre_vv
vh_diff = post_vh - pre_vh

valid = (pre_vv != 0) & (post_vv != 0) & np.isfinite(vv_diff) & np.isfinite(vh_diff)

features = np.stack([pre_vv, pre_vh, post_vv, post_vh, vv_diff, vh_diff], axis=-1)
X_all = features[valid]

print(f"Valid pixels: {X_all.shape[0]}")
pred = clf.predict(X_all)
flooded_pixels = (pred == 1).sum()
pct_flooded = flooded_pixels / X_all.shape[0] * 100
print(f"Predicted flooded pixels: {flooded_pixels} ({pct_flooded:.2f}% of valid area)")

full_pred = np.zeros(shape, dtype=np.uint8)
full_pred[valid] = pred

out_path = os.path.join(OUT_DIR, "predicted_flood_hyderabad.tif")
profile.update(count=1, dtype=np.uint8)
with rasterio.open(out_path, "w", **profile) as dst:
    dst.write(full_pred, 1)
print(f"Saved predicted flood map: {out_path}")

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(np.where(valid, full_pred, np.nan), cmap="Blues", vmin=0, vmax=1)
ax.set_title(
    f"Hyderabad, Musi River flooding (Sept 2024)\n"
    f"Predicted with model trained on Hawke's Bay, NZ - no retraining\n"
    f"{pct_flooded:.1f}% of valid area predicted flooded"
)
ax.axis("off")
fig.tight_layout()
fig_path = os.path.join(OUT_DIR, "predicted_flood_hyderabad.png")
fig.savefig(fig_path, dpi=150)
print(f"Saved figure: {fig_path}")
