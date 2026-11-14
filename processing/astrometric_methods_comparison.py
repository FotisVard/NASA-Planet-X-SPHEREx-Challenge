
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from scipy.ndimage import shift as ndi_shift
from scipy.optimize import minimize

from fits_loader import load_spherex_fits
from image_comparison import compare_images


def flux_weighted_centroid(
    image,
    valid_mask,
    x,
    y,
    radius=6,
):
    """
    Estimate a background-subtracted flux centroid.

    Returns None when the source is not suitable.
    """

    height, width = image.shape

    x = int(round(x))
    y = int(round(y))

    x0 = max(0, x - radius)
    x1 = min(width, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(height, y + radius + 1)

    cutout = image[y0:y1, x0:x1]

    mask = (
        valid_mask[y0:y1, x0:x1]
        & np.isfinite(cutout)
    )

    if np.count_nonzero(mask) < 20:
        return None

    yy, xx = np.indices(cutout.shape)

    # Estimate the local background using border pixels.
    border = np.zeros(cutout.shape, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    background_pixels = cutout[mask & border]

    if len(background_pixels) < 8:
        return None

    background = np.median(background_pixels)

    flux = np.where(
        mask,
        np.maximum(cutout - background, 0),
        0.0,
    )

    total_flux = np.sum(flux)

    if total_flux <= 0:
        return None

    centroid_x = np.sum(
        flux * (xx + x0)
    ) / total_flux

    centroid_y = np.sum(
        flux * (yy + y0)
    ) / total_flux

    return float(centroid_x), float(centroid_y)


def estimate_shift_cross_correlation(
    image1,
    image2,
    valid_mask,
    x,
    y,
    radius=8,
):
    """
    Estimate the relative image displacement using
    normalized correlation and subpixel optimization.

    The returned dx, dy describe the displacement
    of image2 relative to image1.
    """

    height, width = image1.shape

    x = int(round(x))
    y = int(round(y))

    if (
        x - radius < 0
        or y - radius < 0
        or x + radius >= width
        or y + radius >= height
    ):
        return None

    region1 = image1[
        y - radius:y + radius + 1,
        x - radius:x + radius + 1,
    ]

    region2 = image2[
        y - radius:y + radius + 1,
        x - radius:x + radius + 1,
    ]

    region_mask = valid_mask[
        y - radius:y + radius + 1,
        x - radius:x + radius + 1,
    ]

    if np.mean(region_mask) < 0.95:
        return None

    if not (
        np.all(np.isfinite(region1[region_mask]))
        and np.all(np.isfinite(region2[region_mask]))
    ):
        return None

    background1 = np.median(region1[region_mask])
    background2 = np.median(region2[region_mask])

    a = np.where(
        region_mask,
        region1 - background1,
        0.0,
    )

    b = np.where(
        region_mask,
        region2 - background2,
        0.0,
    )

    # Suppress outer pixels to reduce edge effects.
    window_1d = np.hanning(a.shape[0])
    window = np.outer(window_1d, window_1d)

    a = a * window
    b = b * window

    def objective(params):
        dy, dx = params

        shifted_b = ndi_shift(
            b,
            shift=(dy, dx),
            order=3,
            mode="constant",
            cval=0.0,
            prefilter=True,
        )

        norm = (
            np.linalg.norm(a)
            * np.linalg.norm(shifted_b)
        )

        if norm <= 0:
            return 1.0

        correlation = (
            np.sum(a * shifted_b) / norm
        )

        return -float(correlation)

    result = minimize(
        objective,
        x0=[0.0, 0.0],
        method="Powell",
        bounds=[
            (-2.0, 2.0),
            (-2.0, 2.0),
        ],
        options={
            "xtol": 1e-4,
            "ftol": 1e-6,
            "maxiter": 100,
        },
    )

    if not result.success:
        return None

    # The optimization shifts image2 onto image1.
    # Invert the sign to report image2 displacement.
    alignment_dy, alignment_dx = result.x

    dx = -float(alignment_dx)
    dy = -float(alignment_dy)

    return {
        "dx": dx,
        "dy": dy,
        "displacement": float(np.hypot(dx, dy)),
        "correlation": -float(result.fun),
    }


if __name__ == "__main__":

    base_dir = Path(__file__).resolve().parent.parent

    raw_dir = base_dir / "data" / "raw"

    processed_dir = (
        base_dir / "data" / "processed" / "automatic"
    )

    print("\nSPHEREx Astrometric Methods Comparison")
    print("=====================================")

    file1 = raw_dir / (
        "level2_2025W19_1B_0181_3D5_"
        "spx_l2b-v27-2026-197.fits"
    )

    file2 = raw_dir / (
        "level2_2026W20_1A_1166_1D5_"
        "spx_l2b-v27-2026-246.fits"
    )

    print("\nLoading observations...")

    data1 = load_spherex_fits(file1)
    data2 = load_spherex_fits(file2)

    print("\nAligning observations...")

    comparison = compare_images(data1, data2)

    image1 = comparison["image1"]
    image2 = comparison["aligned_image2"]

    with np.load(
        processed_dir / "D5_psf_matching.npz"
    ) as data:
        common_mask = data["common_mask"].astype(bool)

    valid_mask = (
        common_mask
        & np.isfinite(image1)
        & np.isfinite(image2)
    )

    candidates = Table.read(
        processed_dir / "D5_astrometric_results.csv",
        format="csv",
    )

    results = []

    print("\nComparing astrometric methods...")

    for candidate in candidates:

        pair_id = int(candidate["pair_id"])

        x = 0.5 * (
            float(candidate["negative_x"])
            + float(candidate["positive_x"])
        )

        y = 0.5 * (
            float(candidate["negative_y"])
            + float(candidate["positive_y"])
        )

        centroid1 = flux_weighted_centroid(
            image1, valid_mask, x, y
        )

        centroid2 = flux_weighted_centroid(
            image2, valid_mask, x, y
        )

        if centroid1 is not None and centroid2 is not None:
            flux_dx = centroid2[0] - centroid1[0]
            flux_dy = centroid2[1] - centroid1[1]
            flux_displacement = float(
                np.hypot(flux_dx, flux_dy)
            )
        else:
            flux_dx = np.nan
            flux_dy = np.nan
            flux_displacement = np.nan

        correlation = estimate_shift_cross_correlation(
            image1,
            image2,
            valid_mask,
            x,
            y,
        )

        if correlation is not None:
            corr_dx = correlation["dx"]
            corr_dy = correlation["dy"]
            corr_displacement = correlation["displacement"]
            corr_quality = correlation["correlation"]
        else:
            corr_dx = np.nan
            corr_dy = np.nan
            corr_displacement = np.nan
            corr_quality = np.nan

        gaussian_displacement = float(
            candidate["displacement_pixels"]
        )

        results.append({
            "pair_id": pair_id,
            "gaussian_displacement": gaussian_displacement,
            "flux_dx": flux_dx,
            "flux_dy": flux_dy,
            "flux_displacement": flux_displacement,
            "correlation_dx": corr_dx,
            "correlation_dy": corr_dy,
            "correlation_displacement": corr_displacement,
            "correlation_quality": corr_quality,
        })

        print(
            f"Pair {pair_id}: "
            f"Gaussian={gaussian_displacement:.4f}px | "
            f"Flux={flux_displacement:.4f}px | "
            f"Correlation={corr_displacement:.4f}px"
        )

    output_table = Table(rows=results)

    output_csv = (
        processed_dir / "D5_astrometric_methods.csv"
    )

    output_table.write(
        output_csv,
        format="csv",
        overwrite=True,
    )

    pair_ids = [
        row["pair_id"]
        for row in results
    ]

    x_positions = np.arange(len(pair_ids))

    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.bar(
        x_positions - width,
        [row["gaussian_displacement"] for row in results],
        width,
        label="Gaussian",
    )

    ax.bar(
        x_positions,
        [row["flux_displacement"] for row in results],
        width,
        label="Flux centroid",
    )

    ax.bar(
        x_positions + width,
        [row["correlation_displacement"] for row in results],
        width,
        label="Cross-correlation",
    )

    ax.set_xticks(x_positions)
    ax.set_xticklabels([
        f"Pair {i}" for i in pair_ids
    ])

    ax.set_ylabel("Displacement (pixels)")
    ax.set_title("SPHEREx-D5: Astrometric Method Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    plt.tight_layout()

    output_figure = (
        processed_dir / "D5_astrometric_methods.png"
    )

    plt.savefig(
        output_figure,
        dpi=150,
    )

    plt.close(fig)

    print("\nSaved:", output_csv)
    print("Saved:", output_figure)

    print("\nAstrometric methods comparison completed.")
