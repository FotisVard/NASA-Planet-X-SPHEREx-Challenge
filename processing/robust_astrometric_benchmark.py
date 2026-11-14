
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from scipy.ndimage import shift as ndi_shift
from scipy.ndimage import gaussian_filter

from fits_loader import load_spherex_fits
from astrometric_analysis import fit_source
from astrometric_methods_comparison import (
    flux_weighted_centroid,
    estimate_shift_cross_correlation,
)


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "automatic"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SHIFTS = [0.0, 0.05, 0.10, 0.20, 0.50]
ANGLES = [0, 90, 180, 270]

RADIUS = 16
MIN_VALID_FRACTION = 0.90
CENTRAL_RADIUS = 8

NOISE_FRACTION = 0.02
PSF_SIGMA_CHANGE = 0.15

RNG = np.random.default_rng(42)


def extract_cutout(image, mask, x, y):
    x = int(round(x))
    y = int(round(y))

    if (
        x - RADIUS < 0
        or y - RADIUS < 0
        or x + RADIUS >= image.shape[1]
        or y + RADIUS >= image.shape[0]
    ):
        return None, "image_edge"

    cutout = image[
        y - RADIUS:y + RADIUS + 1,
        x - RADIUS:x + RADIUS + 1,
    ].astype(float).copy()

    cutout_mask = mask[
        y - RADIUS:y + RADIUS + 1,
        x - RADIUS:x + RADIUS + 1,
    ]

    valid = cutout_mask & np.isfinite(cutout)

    if np.mean(valid) < MIN_VALID_FRACTION:
        return None, "low_valid_fraction"

    center = RADIUS

    central = valid[
        center - CENTRAL_RADIUS:center + CENTRAL_RADIUS + 1,
        center - CENTRAL_RADIUS:center + CENTRAL_RADIUS + 1,
    ]

    if not np.all(central):
        return None, "invalid_center"

    background = np.median(cutout[valid])

    cutout[~valid] = background

    return cutout, "accepted"


def estimate_noise_scale(cutout):
    border = np.concatenate([
        cutout[0, :],
        cutout[-1, :],
        cutout[1:-1, 0],
        cutout[1:-1, -1],
    ])

    median = np.median(border)

    mad = np.median(np.abs(border - median))

    robust_sigma = 1.4826 * mad

    return max(float(robust_sigma), 1e-8)


def simulate_observation(cutout, dx, dy):
    """
    Synthetic shift, small PSF variation,
    and independent Gaussian noise.
    """

    shifted = ndi_shift(
        cutout,
        shift=(dy, dx),
        order=3,
        mode="nearest",
        prefilter=True,
    )

    shifted = gaussian_filter(
        shifted,
        sigma=PSF_SIGMA_CHANGE,
    )

    noise_sigma = estimate_noise_scale(cutout)

    noise1 = RNG.normal(
        0,
        NOISE_FRACTION * noise_sigma,
        cutout.shape,
    )

    noise2 = RNG.normal(
        0,
        NOISE_FRACTION * noise_sigma,
        cutout.shape,
    )

    image1 = cutout + noise1
    image2 = shifted + noise2

    return image1, image2


def measure_gaussian(image1, image2):
    mask = np.ones(image1.shape, dtype=bool)

    center = RADIUS

    fit1 = fit_source(
        image1, mask, center, center, radius=6
    )

    fit2 = fit_source(
        image2, mask, center, center, radius=6
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


def measure_flux(image1, image2):
    mask = np.ones(image1.shape, dtype=bool)

    center = RADIUS

    c1 = flux_weighted_centroid(
        image1, mask, center, center, radius=6
    )

    c2 = flux_weighted_centroid(
        image2, mask, center, center, radius=6
    )

    if c1 is None or c2 is None:
        return None

    return c2[0] - c1[0], c2[1] - c1[1]


def measure_correlation(image1, image2):
    mask = np.ones(image1.shape, dtype=bool)

    result = estimate_shift_cross_correlation(
        image1,
        image2,
        mask,
        RADIUS,
        RADIUS,
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
    print("\nSPHEREx Robust Astrometric Benchmark")
    print("===================================")

    file1 = RAW_DIR / (
        "level2_2025W19_1B_0181_3D5_"
        "spx_l2b-v27-2026-197.fits"
    )

    data = load_spherex_fits(file1)

    # Reuse the previously aligned image representation
    # if the FITS loader provides it.
    from image_comparison import compare_images

    file2 = RAW_DIR / (
        "level2_2026W20_1A_1166_1D5_"
        "spx_l2b-v27-2026-246.fits"
    )

    data2 = load_spherex_fits(file2)

    comparison = compare_images(data, data2)

    image = comparison["image1"]

    with np.load(
        OUTPUT_DIR / "D5_psf_matching.npz"
    ) as npz:
        mask = npz["common_mask"].astype(bool)

    mask &= np.isfinite(image)

    references = Table.read(
        OUTPUT_DIR / "D5_reference_astrometry.csv",
        format="csv",
    )

    rejection_counts = {}
    accepted_sources = []
    rows = []

    for source_id, source in enumerate(references):
        x = float(source["x"])
        y = float(source["y"])

        cutout, reason = extract_cutout(
            image, mask, x, y
        )

        rejection_counts[reason] = (
            rejection_counts.get(reason, 0) + 1
        )

        if cutout is None:
            continue

        accepted_sources.append(source_id)

        for displacement in SHIFTS:
            for angle in ANGLES:
                theta = np.deg2rad(angle)

                true_dx = displacement * np.cos(theta)
                true_dy = displacement * np.sin(theta)

                image1, image2 = simulate_observation(
                    cutout, true_dx, true_dy
                )

                for method_name, method in METHODS.items():
                    try:
                        measured = method(image1, image2)
                    except (ValueError, RuntimeError):
                        measured = None

                    if measured is None:
                        dx = np.nan
                        dy = np.nan
                        error = np.nan
                    else:
                        dx, dy = measured
                        error = float(
                            np.hypot(
                                dx - true_dx,
                                dy - true_dy,
                            )
                        )

                    rows.append({
                        "source_id": source_id,
                        "method": method_name,
                        "true_displacement": displacement,
                        "angle": angle,
                        "true_dx": true_dx,
                        "true_dy": true_dy,
                        "measured_dx": dx,
                        "measured_dy": dy,
                        "vector_error": error,
                        "success": bool(np.isfinite(error)),
                    })

    print("\nSource selection:")
    print("-----------------")

    for reason, count in rejection_counts.items():
        print(f"{reason}: {count}")

    print(f"\nAccepted sources: {len(accepted_sources)}")

    if not rows:
        raise RuntimeError(
            "No valid sources for benchmark. "
            "Inspect rejection counts before changing thresholds."
        )

    trials = Table(rows=rows)

    trials.write(
        OUTPUT_DIR / "D5_robust_benchmark_trials.csv",
        format="csv",
        overwrite=True,
    )

    summary_rows = []

    print("\nBenchmark summary:")
    print("------------------")

    for method_name in METHODS:
        for displacement in SHIFTS:
            subset = trials[
                (trials["method"] == method_name)
                & np.isclose(
                    trials["true_displacement"],
                    displacement,
                )
            ]

            errors = np.asarray(
                subset["vector_error"],
                dtype=float,
            )

            successful = errors[np.isfinite(errors)]

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

            recovery = (
                len(successful) / len(subset)
                if len(subset)
                else np.nan
            )

            summary_rows.append({
                "method": method_name,
                "shift": displacement,
                "trials": len(subset),
                "successful": len(successful),
                "recovery_rate": recovery,
                "mae": mae,
                "rmse": rmse,
            })

            print(
                f"{method_name:18s} "
                f"shift={displacement:.2f}px "
                f"RMSE={rmse:.4f}px "
                f"recovery={recovery:.1%}"
            )

    summary = Table(rows=summary_rows)

    summary.write(
        OUTPUT_DIR / "D5_robust_benchmark_summary.csv",
        format="csv",
        overwrite=True,
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    for method_name in METHODS:
        subset = summary[
            summary["method"] == method_name
        ]

        ax.plot(
            subset["shift"],
            subset["rmse"],
            marker="o",
            label=method_name,
        )

    ax.set_xlabel("Injected displacement (pixels)")
    ax.set_ylabel("Vector RMSE (pixels)")
    ax.set_title("SPHEREx-D5 Robust Astrometric Benchmark")
    ax.grid(alpha=0.25)
    ax.legend()

    plt.tight_layout()

    output_png = OUTPUT_DIR / "D5_robust_benchmark.png"

    plt.savefig(output_png, dpi=150)
    plt.close(fig)

    print("\nSaved:", output_png)
    print("Robust benchmark completed.")


if __name__ == "__main__":
    main()
