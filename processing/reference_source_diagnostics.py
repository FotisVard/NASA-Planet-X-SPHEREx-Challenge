
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table

from fits_loader import load_spherex_fits
from image_comparison import compare_images


BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"

OUTPUT_DIR = (
    BASE_DIR / "data" / "processed" / "automatic"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CUTOUT_RADIUS = 16


def valid_fraction(valid_mask, x, y, radius):
    """Fraction of valid pixels around a source."""

    x = int(round(x))
    y = int(round(y))

    if (
        x - radius < 0
        or y - radius < 0
        or x + radius >= valid_mask.shape[1]
        or y + radius >= valid_mask.shape[0]
    ):
        return np.nan

    region = valid_mask[
        y - radius:y + radius + 1,
        x - radius:x + radius + 1,
    ]

    return float(np.mean(region))


def nearest_invalid_distance(valid_mask, x, y, radius=16):
    """Distance from source center to nearest invalid pixel."""

    x = int(round(x))
    y = int(round(y))

    if (
        x - radius < 0
        or y - radius < 0
        or x + radius >= valid_mask.shape[1]
        or y + radius >= valid_mask.shape[0]
    ):
        return np.nan

    region = valid_mask[
        y - radius:y + radius + 1,
        x - radius:x + radius + 1,
    ]

    invalid_y, invalid_x = np.where(~region)

    if len(invalid_x) == 0:
        return np.inf

    distances = np.hypot(
        invalid_x - radius,
        invalid_y - radius,
    )

    return float(np.min(distances))


def source_peak(image, x, y, radius=3):
    """Estimate local peak above a simple background."""

    x = int(round(x))
    y = int(round(y))

    if (
        x - radius < 0
        or y - radius < 0
        or x + radius >= image.shape[1]
        or y + radius >= image.shape[0]
    ):
        return np.nan

    region = image[
        y - radius:y + radius + 1,
        x - radius:x + radius + 1,
    ]

    finite = region[np.isfinite(region)]

    if len(finite) == 0:
        return np.nan

    return float(
        np.max(finite) - np.median(finite)
    )


def main():

    print("\nSPHEREx Reference Source Diagnostics")
    print("===================================")

    file1 = RAW_DIR / (
        "level2_2025W19_1B_0181_3D5_"
        "spx_l2b-v27-2026-197.fits"
    )

    file2 = RAW_DIR / (
        "level2_2026W20_1A_1166_1D5_"
        "spx_l2b-v27-2026-246.fits"
    )

    print("\nLoading observations...")

    data1 = load_spherex_fits(file1)
    data2 = load_spherex_fits(file2)

    comparison = compare_images(data1, data2)

    image = comparison["image1"]

    with np.load(
        OUTPUT_DIR / "D5_psf_matching.npz"
    ) as npz:
        common_mask = npz["common_mask"].astype(bool)

    valid_mask = (
        common_mask
        & np.isfinite(image)
        & np.isfinite(comparison["aligned_image2"])
    )

    references = Table.read(
        OUTPUT_DIR / "D5_reference_astrometry.csv",
        format="csv",
    )

    rows = []

    for source_id, source in enumerate(references):

        x = float(source["x"])
        y = float(source["y"])

        fraction_33 = valid_fraction(
            valid_mask, x, y, 16
        )

        fraction_17 = valid_fraction(
            valid_mask, x, y, 8
        )

        fraction_9 = valid_fraction(
            valid_mask, x, y, 4
        )

        fraction_5 = valid_fraction(
            valid_mask, x, y, 2
        )

        nearest_invalid = nearest_invalid_distance(
            valid_mask, x, y
        )

        peak = source_peak(image, x, y)

        if not np.isfinite(fraction_33):
            status = "image_edge"

        elif fraction_33 < 0.90:
            status = "low_valid_fraction"

        elif fraction_17 < 1.0:
            status = "invalid_center"

        else:
            status = "accepted"

        rows.append({
            "source_id": source_id,
            "x": x,
            "y": y,
            "fraction_33": fraction_33,
            "fraction_17": fraction_17,
            "fraction_9": fraction_9,
            "fraction_5": fraction_5,
            "nearest_invalid_distance": nearest_invalid,
            "source_peak": peak,
            "status": status,
        })

    results = Table(rows=rows)

    output_csv = (
        OUTPUT_DIR / "D5_reference_diagnostics.csv"
    )

    results.write(
        output_csv,
        format="csv",
        overwrite=True,
    )

    print("\nSource classification:")
    print("----------------------")

    statuses = np.asarray(results["status"])

    for status in [
        "accepted",
        "invalid_center",
        "low_valid_fraction",
        "image_edge",
    ]:
        count = np.sum(statuses == status)
        print(f"{status}: {count}")

    print("\nValid pixel statistics:")
    print("-----------------------")

    for column in [
        "fraction_33",
        "fraction_17",
        "fraction_9",
        "fraction_5",
    ]:

        values = np.asarray(
            results[column],
            dtype=float,
        )

        values = values[np.isfinite(values)]

        if len(values) == 0:
            continue

        print(
            f"{column}: "
            f"median={np.median(values):.4f}, "
            f"min={np.min(values):.4f}, "
            f"max={np.max(values):.4f}"
        )

    # Figure 1: valid pixel fractions
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5)
    )

    for column, label in [
        ("fraction_33", "33x33"),
        ("fraction_17", "17x17"),
        ("fraction_9", "9x9"),
        ("fraction_5", "5x5"),
    ]:

        values = np.asarray(
            results[column],
            dtype=float,
        )

        values = values[np.isfinite(values)]

        axes[0].hist(
            values,
            bins=30,
            alpha=0.5,
            label=label,
        )

    axes[0].set_xlabel("Valid pixel fraction")
    axes[0].set_ylabel("Number of sources")
    axes[0].set_title("Valid Pixel Fractions")
    axes[0].legend()
    axes[0].grid(alpha=0.2)

    distances = np.asarray(
        results["nearest_invalid_distance"],
        dtype=float,
    )

    distances = distances[np.isfinite(distances)]

    axes[1].hist(
        distances,
        bins=25,
    )

    axes[1].set_xlabel(
        "Nearest invalid pixel distance (px)"
    )

    axes[1].set_ylabel("Number of sources")
    axes[1].set_title("Distance to Invalid Pixels")
    axes[1].grid(alpha=0.2)

    plt.tight_layout()

    output_png = (
        OUTPUT_DIR / "D5_reference_diagnostics.png"
    )

    plt.savefig(output_png, dpi=150)
    plt.close(fig)

    # Figure 2: spatial distribution
    fig, ax = plt.subplots(figsize=(9, 8))

    colors = {
        "accepted": "green",
        "invalid_center": "orange",
        "low_valid_fraction": "red",
        "image_edge": "gray",
    }

    for status, color in colors.items():

        subset = results[
            results["status"] == status
        ]

        if len(subset) == 0:
            continue

        ax.scatter(
            subset["x"],
            subset["y"],
            s=22,
            alpha=0.7,
            color=color,
            label=status,
        )

    ax.set_xlabel("Image X (pixels)")
    ax.set_ylabel("Image Y (pixels)")
    ax.set_title(
        "SPHEREx-D5 Reference Source Quality Map"
    )

    ax.invert_yaxis()
    ax.legend()
    ax.grid(alpha=0.2)

    plt.tight_layout()

    output_map = (
        OUTPUT_DIR / "D5_reference_quality_map.png"
    )

    plt.savefig(output_map, dpi=150)
    plt.close(fig)

    print("\nSaved:", output_csv)
    print("Saved:", output_png)
    print("Saved:", output_map)

    print("\nReference source diagnostics completed.")


if __name__ == "__main__":
    main()
