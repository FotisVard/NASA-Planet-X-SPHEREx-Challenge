
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from scipy.ndimage import shift as ndi_shift

from fits_loader import load_spherex_fits
from image_comparison import compare_images
from astrometric_analysis import fit_source
from astrometric_methods_comparison import (
    flux_weighted_centroid,
    estimate_shift_cross_correlation,
)


BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"

OUTPUT_DIR = (
    BASE_DIR / "data" / "processed" / "automatic"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SHIFTS = [0.0, 0.05, 0.10, 0.20, 0.50]

# Test multiple displacement directions.
ANGLES_DEGREES = [0, 45, 90, 135, 180, 225, 270, 315]

CUTOUT_RADIUS = 16



def extract_cutout(image, mask, x, y, radius=16):
    x = int(round(x))
    y = int(round(y))

    if (
        x - radius < 0
        or y - radius < 0
        or x + radius >= image.shape[1]
        or y + radius >= image.shape[0]
    ):
        return None

    cutout = image[
        y - radius:y + radius + 1,
        x - radius:x + radius + 1,
    ].astype(float).copy()

    cutout_mask = mask[
        y - radius:y + radius + 1,
        x - radius:x + radius + 1,
    ]

    valid = cutout_mask & np.isfinite(cutout)

    # Require at least 95% valid pixels.
    valid_fraction = np.mean(valid)

    if valid_fraction < 0.95:
        return None

    # Keep the central region fully valid.
    center = radius

    central_mask = valid[
        center - 8:center + 9,
        center - 8:center + 9,
    ]

    if not np.all(central_mask):
        return None

    # Fill missing outer pixels with the local median.
    # This is only for the synthetic-shift experiment.
    background = np.median(cutout[valid])

    cutout[~valid] = background

    return cutout

def create_shifted_image(image, dx, dy):
    """
    Artificial subpixel displacement.

    Positive dx shifts the source to the right.
    Positive dy shifts the source downward.
    """

    return ndi_shift(
        image,
        shift=(dy, dx),
        order=3,
        mode="nearest",
        prefilter=True,
    )


def measure_gaussian(image1, image2, x, y):
    mask = np.ones(image1.shape, dtype=bool)

    fit1 = fit_source(
        image1,
        mask,
        x,
        y,
        radius=6,
    )

    fit2 = fit_source(
        image2,
        mask,
        x,
        y,
        radius=6,
    )

    if fit1 is None or fit2 is None:
        return None

    if (
        fit1["at_position_bound"]
        or fit2["at_position_bound"]
    ):
        return None

    return (
        fit2["x"] - fit1["x"],
        fit2["y"] - fit1["y"],
    )


def measure_flux(image1, image2, x, y):
    mask = np.ones(image1.shape, dtype=bool)

    centroid1 = flux_weighted_centroid(
        image1, mask, x, y, radius=6
    )

    centroid2 = flux_weighted_centroid(
        image2, mask, x, y, radius=6
    )

    if centroid1 is None or centroid2 is None:
        return None

    return (
        centroid2[0] - centroid1[0],
        centroid2[1] - centroid1[1],
    )


def measure_correlation(image1, image2, x, y):
    mask = np.ones(image1.shape, dtype=bool)

    result = estimate_shift_cross_correlation(
        image1,
        image2,
        mask,
        x,
        y,
        radius=8,
    )

    if result is None:
        return None

    return result["dx"], result["dy"]


METHODS = {
    "Gaussian": measure_gaussian,
    "Flux centroid": measure_flux,
    "Cross-correlation": measure_correlation,
}


def main():
    print("\nSPHEREx Astrometric Method Validation")
    print("====================================")

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
    ) as data:
        valid_mask = data["common_mask"].astype(bool)

    valid_mask &= np.isfinite(image)

    references = Table.read(
        OUTPUT_DIR / "D5_reference_astrometry.csv",
        format="csv",
    )

    rows = []

    print("\nTesting artificial displacements...")

    total_cutouts = 0

    for reference in references:

        x = float(reference["x"])
        y = float(reference["y"])

        cutout = extract_cutout(
            image,
            valid_mask,
            x,
            y,
            radius=CUTOUT_RADIUS,
        )

        if cutout is None:
            continue

        total_cutouts += 1

        if total_cutouts <= 5:
            print(
                f"Accepted cutout {total_cutouts}: "
                f"x={x:.2f}, y={y:.2f}"
            )

        center = CUTOUT_RADIUS

        for displacement in SHIFTS:

            for angle in ANGLES_DEGREES:

                radians = np.deg2rad(angle)

                true_dx = (
                    displacement * np.cos(radians)
                )

                true_dy = (
                    displacement * np.sin(radians)
                )

                shifted = create_shifted_image(
                    cutout,
                    true_dx,
                    true_dy,
                )

                for method_name, method in METHODS.items():

                    try:
                        measured = method(
                            cutout,
                            shifted,
                            center,
                            center,
                        )
                    except (ValueError, RuntimeError):
                        measured = None

                    if measured is None:
                        rows.append({
                            "method": method_name,
                            "true_displacement": displacement,
                            "angle_degrees": angle,
                            "true_dx": true_dx,
                            "true_dy": true_dy,
                            "measured_dx": np.nan,
                            "measured_dy": np.nan,
                            "vector_error": np.nan,
                            "success": False,
                        })
                        continue

                    measured_dx, measured_dy = measured

                    vector_error = np.hypot(
                        measured_dx - true_dx,
                        measured_dy - true_dy,
                    )

                    rows.append({
                        "method": method_name,
                        "true_displacement": displacement,
                        "angle_degrees": angle,
                        "true_dx": true_dx,
                        "true_dy": true_dy,
                        "measured_dx": measured_dx,
                        "measured_dy": measured_dy,
                        "vector_error": float(vector_error),
                        "success": bool(
                            np.isfinite(vector_error)
                        ),
                    })

    if not rows:
        raise RuntimeError(
            "No valid cutouts were available for testing."
        )

    results = Table(rows=rows)

    output_csv = (
        OUTPUT_DIR / "D5_method_validation_trials.csv"
    )

    results.write(
        output_csv,
        format="csv",
        overwrite=True,
    )

    print("\nValid cutouts:", total_cutouts)

    print("\nValidation Summary")
    print("------------------")

    summary_rows = []

    for method_name in METHODS:

        for displacement in SHIFTS:

            selected = (
                (results["method"] == method_name)
                & np.isclose(
                    results["true_displacement"],
                    displacement,
                )
            )

            subset = results[selected]

            errors = np.asarray(
                subset["vector_error"],
                dtype=float,
            )

            finite = np.isfinite(errors)

            successful = errors[finite]

            total = len(subset)

            recovery_rate = (
                len(successful) / total
                if total > 0
                else np.nan
            )

            mae = (
                float(np.mean(successful))
                if len(successful)
                else np.nan
            )

            rmse = (
                float(np.sqrt(np.mean(successful ** 2)))
                if len(successful)
                else np.nan
            )

            summary_rows.append({
                "method": method_name,
                "true_displacement": displacement,
                "trials": total,
                "successful": len(successful),
                "recovery_rate": recovery_rate,
                "mae": mae,
                "rmse": rmse,
            })

            print(
                f"{method_name:18s} "
                f"shift={displacement:.2f}px "
                f"MAE={mae:.4f}px "
                f"RMSE={rmse:.4f}px "
                f"recovery={recovery_rate:.1%}"
            )

    summary = Table(rows=summary_rows)

    summary_csv = (
        OUTPUT_DIR / "D5_method_validation_summary.csv"
    )

    summary.write(
        summary_csv,
        format="csv",
        overwrite=True,
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    for method_name in METHODS:

        subset = summary[
            summary["method"] == method_name
        ]

        ax.plot(
            subset["true_displacement"],
            subset["rmse"],
            marker="o",
            label=method_name,
        )

    ax.set_xlabel("Injected displacement (pixels)")
    ax.set_ylabel("Vector RMSE (pixels)")
    ax.set_title(
        "SPHEREx-D5: Synthetic Shift Recovery"
    )

    ax.legend()
    ax.grid(alpha=0.25)

    plt.tight_layout()

    output_png = (
        OUTPUT_DIR / "D5_method_validation.png"
    )

    plt.savefig(
        output_png,
        dpi=150,
    )

    plt.close(fig)

    print("\nSaved:", output_csv)
    print("Saved:", summary_csv)
    print("Saved:", output_png)

    print("\nAstrometric method validation completed.")


if __name__ == "__main__":
    main()
