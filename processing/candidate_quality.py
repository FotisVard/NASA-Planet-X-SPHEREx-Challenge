
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from astropy.table import Table
from scipy.ndimage import label

from fits_loader import load_spherex_fits
from image_comparison import compare_images


def get_cutout(image, x, y, radius=8):
    """Return cutout and its valid coordinate bounds."""

    height, width = image.shape

    x0 = max(0, x - radius)
    x1 = min(width, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(height, y + radius + 1)

    return image[y0:y1, x0:x1]


def robust_sigma(values):
    """Median absolute deviation scaled to sigma."""

    values = np.asarray(values)
    values = values[np.isfinite(values)]

    if len(values) < 5:
        return np.nan

    median = np.median(values)

    return float(
        1.4826 * np.median(np.abs(values - median))
    )


def assess_candidate(
    image1,
    image2,
    difference,
    valid_mask,
    candidate,
    radius=8,
):
    x = int(candidate["x"])
    y = int(candidate["y"])

    cut1 = get_cutout(image1, x, y, radius)
    cut2 = get_cutout(image2, x, y, radius)
    cut_diff = get_cutout(difference, x, y, radius)
    cut_mask = get_cutout(valid_mask, x, y, radius)

    valid = (
        cut_mask.astype(bool)
        & np.isfinite(cut1)
        & np.isfinite(cut2)
        & np.isfinite(cut_diff)
    )

    valid_fraction = float(np.mean(valid))

    if np.count_nonzero(valid) < 20:
        return {
            "valid_fraction": valid_fraction,
            "local_scatter": np.nan,
            "local_contrast": np.nan,
            "component_pixels": 0,
            "bright_source": False,
            "quality_flag": "insufficient_data",
        }

    values = cut_diff[valid]

    local_median = float(np.median(values))
    local_scatter = robust_sigma(values)

    center_value = float(difference[y, x])

    if (
        not np.isfinite(local_scatter)
        or local_scatter <= 0
    ):
        local_contrast = np.nan
    else:
        local_contrast = abs(
            center_value - local_median
        ) / local_scatter

    # Estimate brightness relative to local medians.
    source1 = float(image1[y, x])
    source2 = float(image2[y, x])

    background1 = float(np.median(cut1[valid]))
    background2 = float(np.median(cut2[valid]))

    scatter1 = robust_sigma(cut1[valid])
    scatter2 = robust_sigma(cut2[valid])

    bright1 = (
        np.isfinite(scatter1)
        and scatter1 > 0
        and (source1 - background1) > 5 * scatter1
    )

    bright2 = (
        np.isfinite(scatter2)
        and scatter2 > 0
        and (source2 - background2) > 5 * scatter2
    )

    bright_source = bool(bright1 or bright2)

    # Identify connected significant residual pixels.
    if (
        np.isfinite(local_scatter)
        and local_scatter > 0
    ):
        significant = (
            np.abs(cut_diff - local_median)
            > 3 * local_scatter
        ) & valid
    else:
        significant = np.zeros_like(
            valid, dtype=bool
        )

    labeled, _ = label(significant)

    center_y = y - max(0, y - radius)
    center_x = x - max(0, x - radius)

    center_label = labeled[center_y, center_x]

    if center_label > 0:
        component_pixels = int(
            np.count_nonzero(
                labeled == center_label
            )
        )
    else:
        component_pixels = 0

    if valid_fraction < 0.9:
        quality_flag = "low_valid_fraction"
    elif component_pixels <= 1:
        quality_flag = "single_pixel_residual"
    elif bright_source:
        quality_flag = "bright_source_residual"
    else:
        quality_flag = "needs_review"

    return {
        "valid_fraction": valid_fraction,
        "local_scatter": local_scatter,
        "local_contrast": local_contrast,
        "component_pixels": component_pixels,
        "bright_source": bright_source,
        "quality_flag": quality_flag,
    }


if __name__ == "__main__":

    base_dir = Path(__file__).resolve().parent.parent

    raw_dir = base_dir / "data" / "raw"

    processed_dir = (
        base_dir / "data" / "processed" / "automatic"
    )

    candidates = Table.read(
        processed_dir / "D5_candidates_validated.csv",
        format="csv",
    )

    pairs = Table.read(
        base_dir
        / "data"
        / "searches"
        / "spectral_ranked_pairs.csv",
        format="csv",
    )

    matches = [
        row for row in pairs
        if (
            str(row["detector"]) == "SPHEREx-D5"
            and str(row["within_tolerance"]).lower()
            in ("true", "1")
        )
    ]

    if not matches:
        raise RuntimeError(
            "No compatible D5 pair found."
        )

    pair = matches[0]

    filename1 = Path(
        urlparse(str(pair["url_1"])).path
    ).name

    filename2 = Path(
        urlparse(str(pair["url_2"])).path
    ).name

    print("\nSPHEREx Candidate Quality Assessment")
    print("===================================")

    data1 = load_spherex_fits(
        raw_dir / filename1
    )

    data2 = load_spherex_fits(
        raw_dir / filename2
    )

    comparison = compare_images(
        data1,
        data2
    )

    image1 = comparison["image1"]
    image2 = comparison["aligned_image2"]

    with np.load(
        processed_dir / "D5_psf_matching.npz"
    ) as data:
        difference = data[
            "difference_after"
        ].astype(np.float32)

        valid_mask = data[
            "common_mask"
        ].astype(bool)

    results = []

    for candidate in candidates:

        result = assess_candidate(
            image1,
            image2,
            difference,
            valid_mask,
            candidate,
            radius=8,
        )

        results.append(result)

    for key in results[0]:
        candidates[key] = [
            result[key]
            for result in results
        ]

    output_path = (
        processed_dir / "D5_candidate_quality.csv"
    )

    candidates.write(
        output_path,
        format="csv",
        overwrite=True,
    )

    print("\nQuality Assessment Results")
    print("--------------------------")

    flags, counts = np.unique(
        candidates["quality_flag"],
        return_counts=True,
    )

    for flag, count in zip(flags, counts):
        print(f"{flag}: {count}")

    print("\nCandidate details:")

    for candidate in candidates:
        print(
            f"({candidate['x']}, {candidate['y']})",
            f"flag={candidate['quality_flag']}",
            f"pixels={candidate['component_pixels']}",
            f"contrast={candidate['local_contrast']:.2f}",
        )

    print("\nSaved:", output_path)
    print("Candidate quality assessment completed.")
