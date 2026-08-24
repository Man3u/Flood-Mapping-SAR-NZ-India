import os
import glob
import pickle
import numpy as np
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

DATA_DIR = os.path.expanduser("~/Desktop/FloodMappingSAR/data")
OUT_DIR = os.path.expanduser("~/Desktop/FloodMappingSAR/output")
os.makedirs(OUT_DIR, exist_ok=True)

pre_path = os.path.join(DATA_DIR, "hawkesbay_preflood_2023-02-09.tif")
post_path = os.path.join(DATA_DIR, "hawkesbay_postflood_2023-02-14.tif")

ref_candidates = glob.glob(os.path.join(DATA_DIR, "flood_reference", "*.shp"))
if not ref_candidates:
    raise SystemExit("No .shp file found in data/flood_reference/ - check the LINZ download unzipped correctly")
ref_path = ref_candidates[0]
print(f"Using reference flood layer: {ref_path}")

with rasterio.open(pre_path) as src:
    pre_vv = src.read(1).astype(np.float32)
    pre_vh = src.read(2).astype(np.float32)
    transform = src.transform
    crs = src.crs
    shape = src.shape

with rasterio.open(post_path) as src:
    post_vv = src.read(1).astype(np.float32)
    post_vh = src.read(2).astype(np.float32)

vv_diff = post_vv - pre_vv
vh_diff = post_vh - pre_vh

ref = gpd.read_file(ref_path)
ref = ref.to_crs(crs)
flood_mask = rasterize(
    [(geom, 1) for geom in ref.geometry],
    out_shape=shape,
    transform=transform,
    fill=0,
    dtype=np.uint8,
)

valid = (pre_vv != 0) & (post_vv != 0) & np.isfinite(vv_diff) & np.isfinite(vh_diff)

features = np.stack([pre_vv, pre_vh, post_vv, post_vh, vv_diff, vh_diff], axis=-1)
feature_names = ["pre_vv", "pre_vh", "post_vv", "post_vh", "vv_diff", "vh_diff"]

X_all = features[valid]
y_all = flood_mask[valid]

print(f"Valid pixels: {X_all.shape[0]}, flooded: {(y_all==1).sum()}, not flooded: {(y_all==0).sum()}")

rng = np.random.default_rng(42)
flood_idx = np.where(y_all == 1)[0]
noflood_idx = np.where(y_all == 0)[0]
n_sample = min(len(flood_idx), len(noflood_idx), 20000)
flood_sample = rng.choice(flood_idx, n_sample, replace=False)
noflood_sample = rng.choice(noflood_idx, n_sample, replace=False)
sample_idx = np.concatenate([flood_sample, noflood_sample])

X_train_full = X_all[sample_idx]
y_train_full = y_all[sample_idx]

X_train, X_test, y_train, y_test = train_test_split(
    X_train_full, y_train_full, test_size=0.3, random_state=42, stratify=y_train_full
)

clf = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest accuracy: {acc:.4f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred, target_names=["Not flooded", "Flooded"]))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nFeature importance:")
for name, imp in sorted(zip(feature_names, clf.feature_importances_), key=lambda x: -x[1]):
    print(f"  {name}: {imp:.3f}")

full_pred = np.zeros(shape, dtype=np.uint8)
full_pred[valid] = clf.predict(X_all)

out_path = os.path.join(OUT_DIR, "predicted_flood_hawkesbay.tif")
with rasterio.open(pre_path) as src:
    profile = src.profile
profile.update(count=1, dtype=np.uint8)
with rasterio.open(out_path, "w", **profile) as dst:
    dst.write(full_pred, 1)
print(f"\nSaved predicted flood map: {out_path}")

model_path = os.path.join(OUT_DIR, "flood_classifier_model.pkl")
with open(model_path, "wb") as f:
    pickle.dump(clf, f)
print(f"Saved trained model: {model_path}")

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
axes[0].imshow(np.where(valid, flood_mask, np.nan), cmap="Blues", vmin=0, vmax=1)
axes[0].set_title("Reference (LINZ official flood extent)")
axes[0].axis("off")
axes[1].imshow(np.where(valid, full_pred, np.nan), cmap="Blues", vmin=0, vmax=1)
axes[1].set_title(f"Model prediction (Random Forest, {acc:.1%} test accuracy)")
axes[1].axis("off")
fig.suptitle("Hawke's Bay Flood Extent: Reference vs SAR-based Model Prediction (14 Feb 2023)")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "flood_reference_vs_predicted.png"), dpi=150)
print(f"Saved comparison figure: {os.path.join(OUT_DIR, 'flood_reference_vs_predicted.png')}")
