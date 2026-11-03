import os
import OpenEXR
import numpy as np
import cv2

from pathlib import Path
from sklearn.cluster import MiniBatchKMeans

# ============================================================
# SETTINGS
# ============================================================

INPUT_FOLDER = Path(
    "/media/philipm/FOUR/ThesisData/front_notrees/1020/GroundTruth/Color"
)

OUTPUT_FOLDER = Path(
    "/home/philipm/Desktop/imagesTest"
)

# Number of color clusters
N_CLUSTERS = 5

# Fraction of image height where ground starts
GROUND_START = 0.5

# Vegetation detection threshold on normalized ExG
VEGETATION_THRESHOLD = 0.08

# Ignore extremely dark pixels
MIN_BRIGHTNESS = 0.02

# Number of pixels sampled for clustering
SAMPLE_SIZE = 50000

# Save PNG masks for inspection
SAVE_MASKS = True

# Process only the first N images for testing.
# Set to None to process all images.
MAX_IMAGES = 10

RANDOM_STATE = 42

# ============================================================

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

MASK_FOLDER = OUTPUT_FOLDER / "masks"

if SAVE_MASKS:
    MASK_FOLDER.mkdir(parents=True, exist_ok=True)


# ============================================================
# READ EXR
# ============================================================

def read_exr_rgb(path):

    with OpenEXR.File(str(path)) as exr:

        channels = exr.parts[0].channels

        if "RGBA" in channels:
            rgb = channels["RGBA"].pixels[..., :3].astype(
                np.float32
            )

        elif "RGB" in channels:
            rgb = channels["RGB"].pixels.astype(
                np.float32
            )

        elif all(c in channels for c in ("R", "G", "B")):

            rgb = np.stack(
                [
                    channels["R"].pixels,
                    channels["G"].pixels,
                    channels["B"].pixels
                ],
                axis=-1
            ).astype(np.float32)

        else:
            raise ValueError(
                f"Unknown EXR channels: {list(channels.keys())}"
            )

    return rgb


# ============================================================
# NORMALIZED RGB
# ============================================================

def normalize_rgb(rgb):

    # Prevent negative values from affecting color normalization
    rgb_positive = np.maximum(rgb, 0.0)

    total = np.sum(
        rgb_positive,
        axis=-1,
        keepdims=True
    )

    normalized = rgb_positive / np.maximum(total, 1e-8)

    return normalized


# ============================================================
# K-MEANS SEGMENTATION
# ============================================================

def segment_vegetation(rgb):

    height, width = rgb.shape[:2]

    ground_y = int(height * GROUND_START)

    # Only analyze the ground
    ground = rgb[ground_y:, :, :]

    ground_height = ground.shape[0]

    # Normalize RGB
    normalized = normalize_rgb(ground)

    # Flatten pixels
    pixels = normalized.reshape(-1, 3)

    # Original brightness
    brightness = np.mean(
        np.maximum(ground, 0.0),
        axis=2
    ).reshape(-1)

    # Valid pixels
    valid = (
        np.isfinite(pixels).all(axis=1)
        & (brightness > MIN_BRIGHTNESS)
    )

    valid_pixels = pixels[valid]

    if len(valid_pixels) < N_CLUSTERS:
        raise ValueError("Not enough valid pixels for clustering")

    # Randomly sample pixels for faster clustering
    rng = np.random.default_rng(RANDOM_STATE)

    sample_count = min(
        SAMPLE_SIZE,
        len(valid_pixels)
    )

    indices = rng.choice(
        len(valid_pixels),
        sample_count,
        replace=False
    )

    samples = valid_pixels[indices]

    # --------------------------------------------------------
    # FIT K-MEANS
    # --------------------------------------------------------

    kmeans = MiniBatchKMeans(
        n_clusters=N_CLUSTERS,
        random_state=RANDOM_STATE,
        batch_size=4096,
        n_init=5
    )

    kmeans.fit(samples)

    # Predict cluster for every valid ground pixel
    labels = np.full(
        len(pixels),
        -1,
        dtype=np.int32
    )

    labels[valid] = kmeans.predict(valid_pixels)

    labels = labels.reshape(
        ground_height,
        width
    )

    # --------------------------------------------------------
    # IDENTIFY VEGETATION CLUSTERS
    # --------------------------------------------------------

    centers = kmeans.cluster_centers_

    R = centers[:, 0]
    G = centers[:, 1]
    B = centers[:, 2]

    # Normalized Excess Green
    exg = 2 * G - R - B

    vegetation_clusters = np.where(
        exg > VEGETATION_THRESHOLD
    )[0]

    print("\nCluster information:")

    for i in range(N_CLUSTERS):

        classification = (
            "VEGETATION"
            if i in vegetation_clusters
            else "OTHER"
        )

        print(
            f"Cluster {i}: "
            f"RGB={np.round(centers[i], 3)}, "
            f"ExG={exg[i]:.3f}, "
            f"{classification}"
        )

    # --------------------------------------------------------
    # CREATE VEGETATION MASK
    # --------------------------------------------------------

    ground_mask = np.isin(
        labels,
        vegetation_clusters
    )

    # Full-size mask
    full_mask = np.zeros(
        (height, width),
        dtype=bool
    )

    full_mask[ground_y:, :] = ground_mask

    return full_mask


# ============================================================
# SAVE EXR
# ============================================================

def save_exr_rgba(path, rgb, mask):

    # Preserve original RGB for vegetation
    output_rgb = rgb.copy()

    # Remove non-vegetation pixels
    output_rgb[~mask] = 0.0

    # Alpha channel
    alpha = mask.astype(np.float32)

    # Combine RGBA
    rgba = np.dstack(
        (output_rgb, alpha)
    ).astype(np.float32)

    output_exr = OpenEXR.File(
        {},
        {"RGBA": rgba}
    )

    output_exr.write(str(path))


# ============================================================
# MAIN
# ============================================================

exr_files = sorted(
    INPUT_FOLDER.glob("*.exr")
)

if MAX_IMAGES is not None:
    exr_files = exr_files[:MAX_IMAGES]

print(f"Found {len(exr_files)} EXR images.")

for index, image_path in enumerate(exr_files, start=1):

    print(
        f"\n[{index}/{len(exr_files)}] "
        f"Processing {image_path.name}"
    )

    try:

        # Read EXR
        rgb = read_exr_rgb(image_path)

        height, width = rgb.shape[:2]

        # K-means vegetation segmentation
        mask = segment_vegetation(rgb)

        # Save EXR
        output_path = OUTPUT_FOLDER / image_path.name

        save_exr_rgba(
            output_path,
            rgb,
            mask
        )

        # Save mask for inspection
        if SAVE_MASKS:

            mask_image = (
                mask.astype(np.uint8) * 255
            )

            mask_path = (
                MASK_FOLDER /
                f"{image_path.stem}_mask.png"
            )

            cv2.imwrite(
                str(mask_path),
                mask_image
            )

        removed_percentage = (
            100.0 * np.mean(~mask)
        )

        print(
            f"Resolution: {width}x{height}"
        )

        print(
            f"Removed: {removed_percentage:.2f}%"
        )

        print(
            f"Saved: {output_path}"
        )

    except Exception as e:

        print(
            f"ERROR: {image_path.name}: {e}"
        )

print("\nDone!")