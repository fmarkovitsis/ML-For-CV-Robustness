import OpenEXR
import numpy as np
import cv2
import re

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

# Number of K-means clusters
N_CLUSTERS = 8

# Minimum normalized Excess Green for vegetation clusters
VEGETATION_THRESHOLD = 0.08

# Ignore extremely dark pixels
MIN_BRIGHTNESS = 0.02

# Number of pixels sampled for fitting K-means
SAMPLE_SIZE = 50000

# Save vegetation PNG masks
SAVE_MASKS = True

# Save colored K-means cluster visualizations
SAVE_CLUSTER_VISUALIZATIONS = True

# Save visualization for every second image
CLUSTER_VISUALIZATION_INTERVAL = 2

# Add cluster legend to visualization
SHOW_CLUSTER_LEGEND = True

# Set to None to process ALL images
MAX_IMAGES = None

RANDOM_STATE = 42


# ============================================================
# CLUSTER COLORS
# ============================================================

# Distinct RGB colors for visualization only.
# These colors do NOT affect the output EXR pixels.

CLUSTER_COLORS = np.array(
    [
        [31, 119, 180],    # Blue
        [255, 127, 14],    # Orange
        [44, 160, 44],     # Green
        [214, 39, 40],     # Red
        [148, 103, 189],   # Purple
        [140, 86, 75],     # Brown
        [227, 119, 194],   # Pink
        [188, 189, 34],    # Olive
        [23, 190, 207],    # Cyan
        [255, 187, 120],   # Light orange
        [174, 199, 232],   # Light blue
        [152, 223, 138],   # Light green
    ],
    dtype=np.uint8
)


# ============================================================
# FOLDERS
# ============================================================

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

MASK_FOLDER = OUTPUT_FOLDER / "masks"

CLUSTER_FOLDER = (
    OUTPUT_FOLDER / "cluster_visualizations"
)

if SAVE_MASKS:
    MASK_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

if SAVE_CLUSTER_VISUALIZATIONS:
    CLUSTER_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# NATURAL FILE SORTING
# ============================================================

def natural_sort_key(path):

    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", path.name)
    ]


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
# NORMALIZED RGB FOR CLUSTERING ONLY
# ============================================================

def normalize_rgb(rgb):

    rgb_positive = np.maximum(
        rgb,
        0.0
    )

    total = np.sum(
        rgb_positive,
        axis=-1,
        keepdims=True
    )

    normalized = (
        rgb_positive / np.maximum(total, 1e-8)
    )

    return normalized


# ============================================================
# K-MEANS SEGMENTATION
# ============================================================

def segment_vegetation(rgb):

    height, width = rgb.shape[:2]

    # --------------------------------------------------------
    # NORMALIZE RGB FOR K-MEANS ONLY
    # --------------------------------------------------------

    normalized = normalize_rgb(rgb)

    pixels = normalized.reshape(-1, 3)

    # Original brightness
    brightness = np.mean(
        np.maximum(rgb, 0.0),
        axis=2
    ).reshape(-1)

    # --------------------------------------------------------
    # VALID PIXELS
    # --------------------------------------------------------

    valid = (
        np.isfinite(pixels).all(axis=1)
        & np.isfinite(brightness)
        & (brightness > MIN_BRIGHTNESS)
    )

    valid_pixels = pixels[valid]

    if len(valid_pixels) < N_CLUSTERS:

        raise ValueError(
            "Not enough valid pixels for clustering"
        )

    # --------------------------------------------------------
    # SAMPLE PIXELS
    # --------------------------------------------------------

    rng = np.random.default_rng(
        RANDOM_STATE
    )

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

    # --------------------------------------------------------
    # PREDICT EVERY VALID PIXEL
    # --------------------------------------------------------

    labels = np.full(
        len(pixels),
        -1,
        dtype=np.int32
    )

    labels[valid] = kmeans.predict(
        valid_pixels
    )

    labels = labels.reshape(
        height,
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

    # --------------------------------------------------------
    # PRINT CLUSTER INFORMATION
    # --------------------------------------------------------

    print("\nCluster information:")

    for i in range(N_CLUSTERS):

        is_vegetation = (
            i in vegetation_clusters
        )

        classification = (
            "VEGETATION"
            if is_vegetation
            else "OTHER"
        )

        pixel_count = np.sum(
            labels == i
        )

        print(
            f"Cluster {i}: "
            f"RGB={np.round(centers[i], 3)}, "
            f"ExG={exg[i]:.3f}, "
            f"Pixels={pixel_count}, "
            f"{classification}"
        )

    # --------------------------------------------------------
    # CREATE VEGETATION MASK
    # --------------------------------------------------------

    mask = np.isin(
        labels,
        vegetation_clusters
    )

    return (
        mask,
        labels,
        exg,
        vegetation_clusters
    )


# ============================================================
# SAVE EXR WITH ORIGINAL RGB VALUES
# ============================================================

def save_exr_rgba(path, rgb, mask):

    # IMPORTANT:
    # rgb is the ORIGINAL EXR RGB array.
    #
    # No RGB normalization is applied here.
    # No color scaling is applied here.
    # No gamma correction is applied here.

    output_rgb = rgb.copy()

    # Remove non-vegetation pixels
    output_rgb[~mask] = 0.0

    # Vegetation = 1
    # Background = 0
    alpha = mask.astype(
        np.float32
    )

    rgba = np.dstack(
        (output_rgb, alpha)
    ).astype(np.float32)

    output_exr = OpenEXR.File(
        {},
        {"RGBA": rgba}
    )

    output_exr.write(
        str(path)
    )


# ============================================================
# SAVE COLORED K-MEANS CLUSTER VISUALIZATION
# ============================================================

def save_cluster_visualization(
    path,
    labels,
    exg,
    vegetation_clusters
):

    height, width = labels.shape

    # --------------------------------------------------------
    # CREATE COLORED CLUSTER IMAGE
    # --------------------------------------------------------

    cluster_rgb = np.zeros(
        (height, width, 3),
        dtype=np.uint8
    )

    for cluster_id in range(N_CLUSTERS):

        color = CLUSTER_COLORS[
            cluster_id % len(CLUSTER_COLORS)
        ]

        cluster_rgb[
            labels == cluster_id
        ] = color

    # Pixels excluded from K-means remain black.
    # Their label is -1.

    cluster_bgr = cv2.cvtColor(
        cluster_rgb,
        cv2.COLOR_RGB2BGR
    )

    # --------------------------------------------------------
    # OPTIONAL LEGEND
    # --------------------------------------------------------

    if SHOW_CLUSTER_LEGEND:

        legend_width = 370

        legend = np.full(
            (height, legend_width, 3),
            245,
            dtype=np.uint8
        )

        # Scale legend based on image height
        font_scale = max(
            0.4,
            min(0.75, height / 1100)
        )

        thickness = max(
            1,
            round(font_scale * 2)
        )

        row_height = max(
            35,
            int(48 * font_scale / 0.65)
        )

        cv2.putText(
            legend,
            "K-MEANS CLUSTERS",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA
        )

        start_y = 85

        for cluster_id in range(N_CLUSTERS):

            y = start_y + cluster_id * row_height

            color_rgb = CLUSTER_COLORS[
                cluster_id % len(CLUSTER_COLORS)
            ]

            color_bgr = (
                int(color_rgb[2]),
                int(color_rgb[1]),
                int(color_rgb[0])
            )

            is_vegetation = (
                cluster_id in vegetation_clusters
            )

            classification = (
                "VEGETATION"
                if is_vegetation
                else "OTHER"
            )

            # Colored square
            cv2.rectangle(
                legend,
                (20, y - 17),
                (45, y + 8),
                color_bgr,
                -1
            )

            # Cluster label
            text = (
                f"C{cluster_id} "
                f"ExG={exg[cluster_id]:.2f} "
                f"{classification}"
            )

            cv2.putText(
                legend,
                text,
                (60, y + 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale * 0.72,
                (25, 25, 25),
                1,
                cv2.LINE_AA
            )

        # Add legend to right side
        cluster_bgr = np.hstack(
            (cluster_bgr, legend)
        )

    # --------------------------------------------------------
    # SAVE PNG
    # --------------------------------------------------------

    success = cv2.imwrite(
        str(path),
        cluster_bgr
    )

    if not success:
        raise IOError(
            f"Could not save visualization: {path}"
        )


# ============================================================
# MAIN
# ============================================================

exr_files = sorted(
    INPUT_FOLDER.glob("*.exr"),
    key=natural_sort_key
)

total_found = len(exr_files)

if MAX_IMAGES is not None:

    exr_files = exr_files[
        :MAX_IMAGES
    ]

print(
    f"\nInput folder: {INPUT_FOLDER}"
)

print(
    f"Output folder: {OUTPUT_FOLDER}"
)

print(
    f"\nTotal EXR files found: {total_found}"
)

print(
    f"Images selected for processing: {len(exr_files)}"
)


for index, image_path in enumerate(
    exr_files,
    start=1
):

    print(
        f"\n[{index}/{len(exr_files)}] "
        f"Processing {image_path.name}"
    )

    try:

        # ----------------------------------------------------
        # READ ORIGINAL EXR
        # ----------------------------------------------------

        rgb = read_exr_rgb(
            image_path
        )

        height, width = rgb.shape[:2]

        # ----------------------------------------------------
        # SEGMENT VEGETATION
        # ----------------------------------------------------

        (
            mask,
            labels,
            exg,
            vegetation_clusters
        ) = segment_vegetation(rgb)

        # ----------------------------------------------------
        # SAVE VEGETATION EXR
        # ----------------------------------------------------

        output_path = (
            OUTPUT_FOLDER / image_path.name
        )

        save_exr_rgba(
            output_path,
            rgb,
            mask
        )

        # ----------------------------------------------------
        # SAVE VEGETATION MASK
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # SAVE CLUSTER VISUALIZATION
        # EVERY SECOND IMAGE
        # ----------------------------------------------------

        if SAVE_CLUSTER_VISUALIZATIONS:

            if (
                index %
                CLUSTER_VISUALIZATION_INTERVAL
                == 0
            ):

                cluster_path = (
                    CLUSTER_FOLDER /
                    f"{image_path.stem}_clusters.png"
                )

                save_cluster_visualization(
                    cluster_path,
                    labels,
                    exg,
                    vegetation_clusters
                )

                print(
                    f"Cluster visualization: {cluster_path}"
                )

        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        removed_percentage = (
            100.0 * np.mean(~mask)
        )

        retained_percentage = (
            100.0 * np.mean(mask)
        )

        print(
            f"Resolution: {width}x{height}"
        )

        print(
            f"Removed: {removed_percentage:.2f}%"
        )

        print(
            f"Retained: {retained_percentage:.2f}%"
        )

        print(
            f"Saved EXR: {output_path}"
        )

    except Exception as e:

        print(
            f"ERROR processing {image_path.name}: {e}"
        )


print("\nDone!")
