# SAR-Based Flood Mapping: Cyclone Gabrielle (NZ) and Musi River Flooding (India)

A machine learning approach to flood extent mapping using Sentinel-1 SAR imagery, trained on Cyclone Gabrielle's flooding of Hawke's Bay, New Zealand (February 2023), and tested for cross-region generalisation on the Musi River flooding in Hyderabad, India (September 2024).

## Key Finding

A Random Forest classifier trained on Sentinel-1 SAR backscatter (VV, VH, and their pre/post-flood change) achieved 97.4% test accuracy detecting flood extent in Hawke's Bay, validated against LINZ's official Cyclone Gabrielle flood extent layer. An ablation test isolating only the change features (dropping raw pre/post backscatter values) still achieved 79.7% accuracy on its own, confirming the model is genuinely learning from the flood signal, not just memorising static terrain, though the full feature set (which also captures terrain/wetness susceptibility) is what pushes accuracy from the high 70s to the high 90s.

Applying this trained model with no retraining to a Musi River flood event in Hyderabad, India, a different hemisphere, climate, and terrain, the model correctly identified the river channel as flooded, but also produced likely false positives over what appear to be two of Hyderabad's permanent reservoirs (Osman Sagar and Himayat Sagar) and a geometric feature consistent with a runway at Rajiv Gandhi International Airport, which sits at the edge of the study area. This is a known, explainable limitation of SAR-based water detection: smooth surfaces (permanent water, wet pavement, airport tarmac) reflect radar in a similar, water-like way, and a classifier relying partly on raw backscatter values (rather than change alone) cannot fully separate "always wet" from "newly flooded." Reported honestly here rather than masked out, since it's a genuine and instructive limitation of the approach.

## Methodology

**Data source.** Sentinel-1 GRD (IW mode, VV+VH polarisation), accessed via Google Earth Engine. Pre- and post-flood image pairs were selected automatically by searching a date window around each flood event and matching orbit pass (ascending/descending) between the pre and post images, to avoid introducing false backscatter differences from viewing-geometry changes alone.

**Study areas.**
- Hawke's Bay, New Zealand: pre-flood image 9 Feb 2023, post-flood image 14 Feb 2023 (the day Cyclone Gabrielle hit hardest), both ascending pass.
- Hyderabad, India: pre-flood image 25 Aug 2024, post-flood image 6 Sept 2024, covering the Musi River corridor and Old City.

**Preprocessing.** A 5x5 median filter (speckle filter) was applied to all SAR bands before feature extraction. This is standard practice for SAR imagery, which has inherent per-pixel "speckle" noise from the coherent radar imaging process; without it, pixel-by-pixel classification produces widespread scattered false positives unrelated to any real flooding.

**Features.** Six features per pixel: pre-flood VV, pre-flood VH, post-flood VV, post-flood VH, and the VV and VH differences (post minus pre).

**Reference/training data.** Hawke's Bay Regional Council and LINZ published an official validated flood extent polygon layer for Cyclone Gabrielle (14 Feb 2023), sourced from aerial/satellite imagery, ground inspections, and hydrological modelling. This was rasterised to match the SAR grid and used as ground truth for training and testing, a balanced sample of flooded/non-flooded pixels was drawn (20,000 of each class) and split 70/30 for training/testing.

**Model.** Random Forest classifier (200 trees, max depth 15, scikit-learn), trained on the balanced Hawke's Bay sample.

**Ablation test.** To check whether the model was genuinely learning from the flood *change* signal rather than static terrain characteristics, a second model was trained using only the two change features (VV/VH difference), with the raw pre/post values excluded. This dropped accuracy from 97.4% to 79.7%, confirming both a genuine change-detection signal (meaningful on its own) and a substantial contribution from terrain/wetness context in the full model.

**Cross-region transfer test.** The trained Hawke's Bay model (no retraining, no fine-tuning) was applied directly to the Hyderabad SAR pair to test generalisation across a structurally different climate, terrain, and hemisphere.

## Data Sources

| Data | Source | Resolution | Purpose |
|---|---|---|---|
| Sentinel-1 GRD | Copernicus, via Google Earth Engine | 10m | SAR backscatter (VV, VH) |
| Cyclone Gabrielle Flood Areas (14 Feb 2023) | LINZ Data Service | vector | Training/validation ground truth |

## Repository Structure

Raw SAR rasters (`data/*.tif`) and derived prediction rasters are excluded from version control due to file size — fully reproducible by running the scripts in order against fresh Earth Engine exports and the public LINZ layer.

## Tech Stack

Python (rasterio, numpy, scipy, geopandas, scikit-learn, matplotlib), Google Earth Engine, Git/GitHub.

## Author

**Manu Chauhan Mudavath**
MSc Computer Science — University of Waikato, New Zealand
MSc Information Systems — University of West London, UK
BTech — MGIT Hyderabad, India

[LinkedIn](https://linkedin.com/in/manu-chauhan-mudavath) · manuchauhanm76@gmail.com
