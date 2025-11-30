import os
import numpy as np
import pandas as pd
import rasterio

DATA_DIR = r'./data/features'

FEATURES = [
    'DEM.tif',
    'slope.tif',
    'road.tif',
    'ndvi.tif',
    'pop.tif',
    'nightlight.tif',
    'water.tif'
]

MASKS = {
    'positive': r'./data/masks/source.tif',
    'negative': r'./data/masks/low.tif'
}

N_SAMPLES = 10000
RANDOM_SEED =
OUTPUT_CSV = r'./data/samples.csv'

mask_path = MASKS['positive']
with rasterio.open(mask_path) as src:
    print(f"Positive mask size: {src.shape} (height, width)")

dem_path = os.path.join(DATA_DIR, FEATURES[0])
with rasterio.open(dem_path) as src:
    print(f"DEM size: {src.shape} (height, width)")


def check_consistency(mask_path: str, feature_paths: list) -> dict:
    with rasterio.open(mask_path) as src:
        mask_meta = {
            "shape": src.shape,
            "crs": src.crs,
            "transform": src.transform
        }

    for fname in feature_paths:
        with rasterio.open(os.path.join(DATA_DIR, fname)) as src:
            if src.shape != mask_meta["shape"]:
                raise ValueError(f"Feature '{fname}' has a different shape from the mask.")
            if src.crs != mask_meta["crs"]:
                raise ValueError(f"Feature '{fname}' has a different CRS from the mask.")

    return mask_meta


def main():
    print("\n[1/5] Checking data consistency...")
    _ = check_consistency(MASKS['positive'], FEATURES)
    print("Data consistency check passed.")

    print("[2/5] Reading masks and collecting valid pixels...")
    mask_arrays = {}
    for label, path in MASKS.items():
        with rasterio.open(path) as src:
            mask_arrays[label] = src.read(1)

    print("[3/5] Sampling positive and negative pixels...")
    np.random.seed(RANDOM_SEED)
    sample_indices = {}

    for label in ['positive', 'negative']:
        mask = mask_arrays[label]
        valid_idx = np.argwhere(mask == 1)

        if len(valid_idx) < N_SAMPLES:
            raise ValueError(
                f"Mask '{label}' has only {len(valid_idx)} valid pixels, "
                f"but {N_SAMPLES} samples are requested. "
                f"Please adjust the mask or reduce N_SAMPLES."
            )

        sample_indices[label] = valid_idx[
            np.random.choice(len(valid_idx), N_SAMPLES, replace=False)
        ]

    print(f"Successfully sampled: {N_SAMPLES} positive and {N_SAMPLES} negative pixels.")

    print("[4/5] Extracting feature values...")
    feature_arrays = []
    nodata_vals = []

    for fname in FEATURES:
        with rasterio.open(os.path.join(DATA_DIR, fname)) as src:
            feature_arrays.append(src.read(1))
            nodata_vals.append(src.nodata)

    all_records = None

    for label in ['positive', 'negative']:
        rows, cols = sample_indices[label][:, 0], sample_indices[label][:, 1]

        feature_vals = []
        for arr, nodata in zip(feature_arrays, nodata_vals):
            vals = arr[rows, cols]
            if nodata is not None:
                vals = np.where(vals == nodata, np.nan, vals)
            feature_vals.append(vals)

        feature_vals = np.column_stack(feature_vals)
        labels = np.full(N_SAMPLES, 0 if label == 'positive' else 1)
        records = np.column_stack([feature_vals, labels])

        if all_records is None:
            all_records = records
        else:
            all_records = np.concatenate([all_records, records], axis=0)

    print("[5/5] Saving samples to CSV...")
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    columns = [
        'DEM', 'Slope', 'RoadDist', 'NDVI',
        'PopDensity', 'Nightlight', 'WaterDist', 'Label'
    ]

    df = pd.DataFrame(all_records, columns=columns)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"Sample data saved to: {OUTPUT_CSV}")
    print(
        f"Data shape: {df.shape[0]} rows "
        f"({N_SAMPLES} positive + {N_SAMPLES} negative) × {df.shape[1]} columns "
        f"({len(FEATURES)} features + 1 label)."
    )
    print("Preview (first 5 rows):")
    print(df.head())

if __name__ == "__main__":
    main()