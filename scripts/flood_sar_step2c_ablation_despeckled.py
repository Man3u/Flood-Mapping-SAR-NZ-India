import os
import glob
import numpy as np
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import median_filter
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

DATA_DIR = os.path.expanduser("~/Desktop/FloodMappingSAR/data")

pre_path = os.path.join(DATA_DIR, "hawkesbay_preflood_2023-02-09.tif")
post_path = os.path.join(DATA_DIR, "hawkesbay_postflood_2023-02-14.tif")
ref_path = glob.glob(os.path.join(DATA_DIR, "flood_reference", "*.shp"))[0]

with rasterio.open(pre_path) as src:
    pre_vv = median_filter(src.read(1).astype(np.float32), size=5)
    pre_vh = median_filter(src.read(2).astype(np.float32), size=5)
    transform = src.transform
    crs = src.crs
    shape = src.shape

with rasterio.open(post_path) as src:
    post_vv = median_filter(src.read(1).astype(np.float32), size=5)
    post_vh = median_filter(src.read(2).astype(np.float32), size=5)

vv_diff = post_vv - pre_vv
vh_diff = post_vh - pre_vh

ref = gpd.read_file(ref_path).to_crs(crs)
flood_mask = rasterize([(g, 1) for g in ref.geometry], out_shape=shape, transform=transform, fill=0, dtype=np.uint8)

valid = (pre_vv != 0) & (post_vv != 0) & np.isfinite(vv_diff) & np.isfinite(vh_diff)

features = np.stack([vv_diff, vh_diff], axis=-1)
X_all = features[valid]
y_all = flood_mask[valid]

rng = np.random.default_rng(42)
flood_idx = np.where(y_all == 1)[0]
noflood_idx = np.where(y_all == 0)[0]
n_sample = min(len(flood_idx), len(noflood_idx), 20000)
sample_idx = np.concatenate([
    rng.choice(flood_idx, n_sample, replace=False),
    rng.choice(noflood_idx, n_sample, replace=False),
])

X_train, X_test, y_train, y_test = train_test_split(
    X_all[sample_idx], y_all[sample_idx], test_size=0.3, random_state=42, stratify=y_all[sample_idx]
)

clf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(f"Change-only model test accuracy (despeckled): {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=["Not flooded", "Flooded"]))
print(f"Feature importance - vv_diff: {clf.feature_importances_[0]:.3f}, vh_diff: {clf.feature_importances_[1]:.3f}")
