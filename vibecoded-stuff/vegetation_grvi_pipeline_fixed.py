"""One-pass EXR -> vegetation segmentation -> 20 AOIs -> perceived GRVI CSVs.

Replaces remove_soil.py, aoi-no-soil.ipynb, and Untitled.ipynb.

Run on the Linux machine:
    python -m pip install OpenEXR numpy pandas scikit-learn
    python vegetation_grvi_pipeline.py

CSV files written to ~/Desktop:
    perceived_grvi_by_aoi.csv
    perceived_grvi_by_material.csv
    perceived_grvi_by_image.csv
    grvi_image_order.csv

No modified EXR files are needed or written. RGB is read directly from the
original EXR, and the K-means vegetation mask is kept in memory.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import OpenEXR
from sklearn.cluster import MiniBatchKMeans


# =========================== SETTINGS ==============================
ORIGINAL_EXR_FOLDER = Path(
    "/media/philipm/FOUR/ThesisData/front_notrees/1020/GroundTruth/Color"
)
GROUND_TRUTH_CSV = Path(
    "/media/philipm/FOUR/ThesisData/groundtruthGRVI-front.csv"
)
DESKTOP = Path("/home/philipm/Desktop")

# Read directly from ORIGINAL_EXR_FOLDER in deterministic filename order.
# Set IMAGE_ORDER to "legacy" ONLY if imagesTest still contains the original
# segmented EXRs and you specifically need their old glob() enumeration order.
# Confirm that the CSV rows follow the selected image order.
LEGACY_SEGMENTED_FOLDER = Path("/home/philipm/Desktop/imagesTest")
IMAGE_ORDER = "sorted_original"  # "sorted_original" or "legacy"

# Original pipeline's K-means parameters
N_CLUSTERS = 8
VEGETATION_THRESHOLD = 0.08
MIN_BRIGHTNESS = 0.02
SAMPLE_SIZE = 50_000
RANDOM_STATE = 42

# Use each image's own vegetation mask (original behavior).
# 'reference' uses one fixed mask for every image. It reduces color selection
# bias, but is ONLY valid if all images show the SAME pixel-aligned scene.
MASK_MODE = "per_image"          # 'per_image' or 'reference'
REFERENCE_IMAGE = "img_0019.exr"

# AOI geometry is kept exactly as in aoi-no-soil.ipynb
NUMBER_OF_AOIS = 20
GRID_DIVISOR = 24
FIRST_AOI_GRID_COL = 2

# Irradiances from Untitled.ipynb
RED_IRRADIANCE = 1123.904772
GREEN_IRRADIANCE = 1020.232214

# AOIs with fewer than this many usable vegetation pixels become NaN
MIN_VEGETATION_PIXELS = 1

# Restrict to first N images for a smoke test; None processes all.
MAX_IMAGES = None
# ===================================================================


def read_exr_rgb(path: Path) -> np.ndarray:
    """Read RGB from modern OpenEXR Python bindings; copy before closing."""
    with OpenEXR.File(str(path)) as exr:
        channels = exr.parts[0].channels
        if "RGBA" in channels:
            rgb = channels["RGBA"].pixels[..., :3]
        elif "RGB" in channels:
            rgb = channels["RGB"].pixels[..., :3]
        elif all(k in channels for k in ("R", "G", "B")):
            rgb = np.stack(
                [channels["R"].pixels,
                 channels["G"].pixels,
                 channels["B"].pixels],
                axis=-1,
            )
        else:
            raise ValueError(
                f"{path.name} has no RGB(A) channels; found {list(channels)}"
            )
        return np.asarray(rgb, dtype=np.float32).copy()


def vegetation_mask_kmeans(rgb: np.ndarray) -> np.ndarray:
    """Cluster the WHOLE image and retain green-dominant cluster(s)."""
    # Same normalized RGB + green-cluster criterion as remove_soil.py.
    nonnegative = np.maximum(rgb, 0.0)
    total = np.sum(nonnegative, axis=-1, keepdims=True)
    normalized = nonnegative / np.maximum(total, 1e-8)
    flat = normalized.reshape(-1, 3)
    brightness = np.mean(nonnegative, axis=-1).reshape(-1)
    valid = np.isfinite(flat).all(axis=1) & (brightness > MIN_BRIGHTNESS)
    valid_pixels = flat[valid]

    if len(valid_pixels) < N_CLUSTERS:
        raise ValueError("Not enough valid pixels for K-means")

    # Fixed seed for repeatability. Fit on samples; predict on all valid pixels.
    rng = np.random.default_rng(RANDOM_STATE)
    sample_indices = rng.choice(
        len(valid_pixels),
        size=min(SAMPLE_SIZE, len(valid_pixels)),
        replace=False,
    )
    kmeans = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        batch_size=4096,
        n_init=5,
    )
    kmeans.fit(valid_pixels[sample_indices])

    centers = kmeans.cluster_centers_
    exg = 2.0 * centers[:, 1] - centers[:, 0] - centers[:, 2]
    keep_clusters = np.flatnonzero(exg > VEGETATION_THRESHOLD)

    mask_flat = np.zeros(len(flat), dtype=bool)
    if keep_clusters.size:
        labels = kmeans.predict(valid_pixels)
        mask_flat[valid] = np.isin(labels, keep_clusters)
    return mask_flat.reshape(rgb.shape[:2])


def aoi_positions(height: int, width: int):
    """Reproduce the ORIGINAL 20-block horizontal row, without resizing."""
    step = width // GRID_DIVISOR
    if step < 1:
        raise ValueError(f"Image too narrow: {width}")
    x0 = FIRST_AOI_GRID_COL * step
    y0 = height // 2 - step // 2
    if y0 < 0 or y0 + step > height or x0 + NUMBER_OF_AOIS * step > width:
        raise ValueError(f"AOI row extends outside {width}x{height} image")
    for j in range(NUMBER_OF_AOIS):
        x = x0 + j * step
        yield j, x, y0, step


def grvi_stats(rgb_aoi: np.ndarray, selected: np.ndarray | None = None):
    """Mean per-pixel GRVI. Applies a SHARED pixel mask to red and green.

    rho_R = pi * L_R / E_R, rho_G = pi * L_G / E_G.
    pi cancels in (rho_G-rho_R)/(rho_G+rho_R).
    """
    r = rgb_aoi[..., 0].astype(np.float64, copy=False) / RED_IRRADIANCE
    g = rgb_aoi[..., 1].astype(np.float64, copy=False) / GREEN_IRRADIANCE
    denom = r + g
    valid = (np.isfinite(r) & np.isfinite(g) &
             (r >= 0.0) & (g >= 0.0) & (denom > 1e-12))
    if selected is not None:
        valid &= selected
    count = int(np.count_nonzero(valid))
    if count == 0:
        return np.nan, np.nan, count
    values = (g[valid] - r[valid]) / denom[valid]
    return float(np.mean(values)), float(np.std(values)), count


def find_images() -> list[Path]:
    if IMAGE_ORDER == "legacy":
        if not LEGACY_SEGMENTED_FOLDER.is_dir():
            raise FileNotFoundError(
                f"Legacy order folder missing: {LEGACY_SEGMENTED_FOLDER}\n"
                "To reproduce your verified ground-truth order, this folder "
                "must contain the EXRs used by your previous AOI notebook."
            )
        # Intentionally NOT sorted: mirrors aoi-no-soil.ipynb exactly.
        names = [p.name for p in LEGACY_SEGMENTED_FOLDER.glob("*.exr")]
        images = [ORIGINAL_EXR_FOLDER / name for name in names]
    elif IMAGE_ORDER == "sorted_original":
        images = sorted(ORIGINAL_EXR_FOLDER.glob("*.exr"))
    else:
        raise ValueError(f"Unknown IMAGE_ORDER: {IMAGE_ORDER}")

    if not images:
        if IMAGE_ORDER == "legacy":
            raise FileNotFoundError(
                f"No EXR files found in legacy folder: {LEGACY_SEGMENTED_FOLDER}\n"
                "Set IMAGE_ORDER = 'sorted_original' to read from "
                f"{ORIGINAL_EXR_FOLDER} instead."
            )
        raise FileNotFoundError(
            f"No EXR files found in original folder: {ORIGINAL_EXR_FOLDER}"
        )
    if len({p.name for p in images}) != len(images):
        raise ValueError("Duplicate EXR names in processing order")
    missing = [p for p in images if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Original EXR missing for {len(missing)} image(s); "
            f"first: {missing[0]}"
        )
    return images


def summarize_by_material(aoi_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for inherent, group in aoi_df.groupby("grviOut", sort=True):
        used = group.dropna(subset=["grviIn"])
        weights = used["nVegetationPixels"].to_numpy(dtype=np.float64)
        measured = used["grviIn"].to_numpy(dtype=np.float64)
        pixel_weighted = (
            float(np.average(measured, weights=weights))
            if len(used) and weights.sum() > 0 else np.nan
        )
        rows.append({
            "grviOut": inherent,
            "meanPerceivedGRVI": group["grviIn"].mean(),
            "stdPerceivedGRVI": group["grviIn"].std(ddof=1),
            "pixelWeightedMeanPerceivedGRVI": pixel_weighted,
            "meanGRVIBeforeSegmentation": group["grviBeforeSegmentation"].mean(),
            "meanVegetationCoverage": group["vegetationCoverage"].mean(),
            "numberOfImages": group["image"].nunique(),
            "totalAOIs": len(group),
            "validAOIs": len(used),
            "totalVegetationPixels": int(group["nVegetationPixels"].sum()),
        })
    return pd.DataFrame(rows)


def main():
    if RED_IRRADIANCE <= 0 or GREEN_IRRADIANCE <= 0:
        raise ValueError("Irradiances must be positive")
    if MASK_MODE not in {"per_image", "reference"}:
        raise ValueError("MASK_MODE must be 'per_image' or 'reference'")
    if not GROUND_TRUTH_CSV.is_file():
        raise FileNotFoundError(f"Ground-truth CSV missing: {GROUND_TRUTH_CSV}")
    if not ORIGINAL_EXR_FOLDER.is_dir():
        raise FileNotFoundError(f"Original EXR folder missing: {ORIGINAL_EXR_FOLDER}")
    DESKTOP.mkdir(parents=True, exist_ok=True)

    image_paths = find_images()
    gt = pd.read_csv(GROUND_TRUTH_CSV)
    if "grviOut" not in gt.columns:
        raise ValueError("Ground-truth CSV must have a 'grviOut' column")
    inherent_values = pd.to_numeric(gt["grviOut"], errors="raise")
    if inherent_values.isna().any():
        raise ValueError("Ground-truth CSV has blank grviOut values")

    # Important: check FULL image-count-to-CSV correspondence before a test run.
    expected_rows = len(image_paths) * NUMBER_OF_AOIS
    print(f"Found {len(image_paths)} EXRs in {ORIGINAL_EXR_FOLDER}")
    print(f"First 5 selected images: {[p.name for p in image_paths[:5]]}")
    print(f"Last 5 selected images: {[p.name for p in image_paths[-5:]]}")
    print(f"Ground-truth rows: {len(inherent_values)}; "
          f"expected images: {len(inherent_values) / NUMBER_OF_AOIS:g}")
    if len(inherent_values) != expected_rows:
        raise ValueError(
            f"Image/CSV count mismatch: {len(image_paths)} images x "
            f"{NUMBER_OF_AOIS} AOIs = {expected_rows} rows, but CSV has "
            f"{len(inherent_values)} rows. Your original processing may have "
            "used only a subset of EXRs. Check which filenames belong in "
            "the dataset and their order; this script deliberately will "
            "not drop images or assign labels to the wrong files."
        )

    if MAX_IMAGES is not None:
        image_paths = image_paths[:MAX_IMAGES]

    manifest = []
    for i, path in enumerate(image_paths):
        manifest.append({"imageIndex": i, "image": path.name,
                         "gtFirstAOI": float(inherent_values.iloc[i * NUMBER_OF_AOIS])})
    pd.DataFrame(manifest).to_csv(DESKTOP / "grvi_image_order.csv", index=False)

    print(f"Input: {ORIGINAL_EXR_FOLDER}")
    print(f"Ground truth: {GROUND_TRUTH_CSV}")
    print(f"Image order: {IMAGE_ORDER}")
    print(f"Images selected: {len(image_paths)} (full dataset: {expected_rows // NUMBER_OF_AOIS})")
    print(f"Segmentation: {MASK_MODE}")
    print(f"Output: {DESKTOP}")

    reference_mask = None
    if MASK_MODE == "reference":
        reference_path = ORIGINAL_EXR_FOLDER / REFERENCE_IMAGE
        reference_mask = vegetation_mask_kmeans(read_exr_rgb(reference_path))
        print(f"Reference mask from {reference_path.name} (requires pixel alignment)")

    records = []
    for i, path in enumerate(image_paths):
        rgb = read_exr_rgb(path)
        height, width = rgb.shape[:2]
        if MASK_MODE == "reference":
            if reference_mask.shape != (height, width):
                raise ValueError(f"Reference mask shape mismatch for {path.name}")
            mask = reference_mask
        else:
            mask = vegetation_mask_kmeans(rgb)

        for j, x0, y0, step in aoi_positions(height, width):
            rgb_aoi = rgb[y0:y0+step, x0:x0+step]
            mask_aoi = mask[y0:y0+step, x0:x0+step]
            raw_mean, _, raw_n = grvi_stats(rgb_aoi)
            mean, std, n = grvi_stats(rgb_aoi, mask_aoi)
            if n < MIN_VEGETATION_PIXELS:
                mean, std = np.nan, np.nan
            row_index = i * NUMBER_OF_AOIS + j
            records.append({
                "imageIndex": i,
                "image": path.name,
                "aoiIndex": j,
                "x0": x0,
                "y0": y0,
                "aoiSize": step,
                "grviOut": float(inherent_values.iloc[row_index]),
                "grviIn": mean,
                "grviStd": std,
                "grviBeforeSegmentation": raw_mean,  # includes soil/other pixels
                "nVegetationPixels": n,
                "nValidOriginalPixels": raw_n,
                "vegetationCoverage": n / (step * step),
            })
        print(f"[{i+1}/{len(image_paths)}] {path.name}: "
              f"vegetation={100 * mask.mean():.1f}% of full image")

    aoi_df = pd.DataFrame(records)
    aoi_csv = DESKTOP / "perceived_grvi_by_aoi.csv"
    aoi_df.to_csv(aoi_csv, index=False)

    material_df = summarize_by_material(aoi_df)
    material_csv = DESKTOP / "perceived_grvi_by_material.csv"
    material_df.to_csv(material_csv, index=False)

    image_df = aoi_df.groupby(
        ["imageIndex", "image", "grviOut"], sort=False, as_index=False
    ).agg(
        meanPerceivedGRVI=("grviIn", "mean"),
        meanGRVIBeforeSegmentation=("grviBeforeSegmentation", "mean"),
        meanVegetationCoverage=("vegetationCoverage", "mean"),
        validAOIs=("grviIn", "count"),
    )
    image_csv = DESKTOP / "perceived_grvi_by_image.csv"
    image_df.to_csv(image_csv, index=False)

    print("\nMEAN PERCEIVED GRVI PER INHERENT GRVI")
    print(material_df[[
        "grviOut", "meanPerceivedGRVI", "stdPerceivedGRVI",
        "pixelWeightedMeanPerceivedGRVI", "validAOIs", "totalAOIs"
    ]].to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nSaved {len(aoi_df)} AOI rows to: {aoi_csv}")
    print(f"Saved material means to: {material_csv}")
    print(f"Saved image means to: {image_csv}")
    print(f"Saved processing order to: {DESKTOP / 'grvi_image_order.csv'}")
    if aoi_df["grviIn"].isna().any():
        print(f"WARNING: {aoi_df['grviIn'].isna().sum()} AOIs have no valid "
              "vegetation pixels and are stored as NaN.")
    if MASK_MODE == "per_image":
        print("NOTE: per-image green-cluster selection may bias GRVI toward "
              "similar values. Compare the before/after and coverage columns.")


if __name__ == "__main__":
    main()
