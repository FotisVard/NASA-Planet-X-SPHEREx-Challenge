
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from scipy.optimize import least_squares

from fits_loader import load_spherex_fits
from image_comparison import compare_images


def gaussian_2d(params, x, y):
    """Elliptical, axis-aligned 2D Gaussian plus background."""

    amplitude, x0, y0, sigma_x, sigma_y, background = params

    exponent = (
        ((x - x0) ** 2) / (2 * sigma_x ** 2)
        + ((y - y0) ** 2) / (2 * sigma_y ** 2)
    )

    return background + amplitude * np.exp(-exponent)


def fit_source(image, valid_mask, x, y, radius=6):
    """
    Fit a local Gaussian source model.

    The fit position is initialized from the brightest
    pixel within 3 pixels of the supplied position.
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

    global_x = xx + x0
    global_y = yy + y0

    values = cutout[mask]

    background = float(np.median(values))

    # Search for the brightest pixel near the expected
    # position, rather than over the entire cutout.
    search_mask = (
        mask
        & ((global_x - x) ** 2 + (global_y - y) ** 2 <= 9)
    )

    if not np.any(search_mask):
        return None

    search_values = np.where(
        search_mask,
        cutout,
        -np.inf,
    )

    peak_y, peak_x = np.unravel_index(
        np.argmax(search_values),
        search_values.shape,
    )

    initial_x = float(global_x[peak_y, peak_x])
    initial_y = float(global_y[peak_y, peak_x])

    amplitude = float(
        cutout[peak_y, peak_x] - background
    )

    if amplitude <= 0:
        return None

    initial = [
        amplitude,
        initial_x,
        initial_y,
        1.2,
        1.2,
        background,
    ]

    lower_bounds = [
        0.0,
        x - 3.0,
        y - 3.0,
        0.5,
        0.5,
        -np.inf,
    ]

    upper_bounds = [
        np.inf,
        x + 3.0,
        y + 3.0,
        5.0,
        5.0,
        np.inf,
    ]

    # Ensure the initial point satisfies the bounds.
    initial[1] = np.clip(
        initial[1],
        lower_bounds[1] + 1e-6,
        upper_bounds[1] - 1e-6,
    )

    initial[2] = np.clip(
        initial[2],
        lower_bounds[2] + 1e-6,
        upper_bounds[2] - 1e-6,
    )

    def residuals(params):
        model = gaussian_2d(
            params,
            global_x[mask],
            global_y[mask],
        )

        return model - cutout[mask]

    result = least_squares(
        residuals,
        initial,
        bounds=(lower_bounds, upper_bounds),
        max_nfev=300,
    )

    if not result.success:
        return None

    fitted = result.x

    model_values = gaussian_2d(
        fitted,
        global_x[mask],
        global_y[mask],
    )

    fit_residuals = cutout[mask] - model_values

    rms = float(
        np.sqrt(np.mean(fit_residuals ** 2))
    )

    amplitude_fit = float(fitted[0])

    # Heuristic fit contrast, not formal SNR.
    contrast = (
        amplitude_fit / rms
        if rms > 0
        else np.nan
    )

    return {
        "x": float(fitted[1]),
        "y": float(fitted[2]),
        "sigma_x": float(fitted[3]),
        "sigma_y": float(fitted[4]),
        "amplitude": amplitude_fit,
        "background": float(fitted[5]),
        "fit_rms": rms,
        "fit_contrast": contrast,
        "valid_pixels": int(np.count_nonzero(mask)),
        "at_position_bound": bool(
            abs(fitted[1] - (x - 3)) < 0.05
            or abs(fitted[1] - (x + 3)) < 0.05
            or abs(fitted[2] - (y - 3)) < 0.05
            or abs(fitted[2] - (y + 3)) < 0.05
        ),
    }


def analyze_motion_pair(
    image1,
    image2,
    valid_mask,
    pair,
):
    """
    Fit both epochs near the same approximate source
    location and compare their fitted centroids.
    """

    negative_x = int(pair["negative_x"])
    negative_y = int(pair["negative_y"])

    positive_x = int(pair["positive_x"])
    positive_y = int(pair["positive_y"])

    center_x = int(round(
        (negative_x + positive_x) / 2
    ))

    center_y = int(round(
        (negative_y + positive_y) / 2
    ))

    fit1 = fit_source(
        image1,
        valid_mask,
        center_x,
        center_y,
    )

    fit2 = fit_source(
        image2,
        valid_mask,
        center_x,
        center_y,
    )

    if fit1 is None or fit2 is None:
        return {
            "fit_success": False,
            "dx_pixels": np.nan,
            "dy_pixels": np.nan,
            "displacement_pixels": np.nan,
        }

    dx = fit2["x"] - fit1["x"]
    dy = fit2["y"] - fit1["y"]

    displacement = float(
        np.hypot(dx, dy)
    )

    # A fit reaching its position bounds is unreliable.
    fit_success = not (
        fit1["at_position_bound"]
        or fit2["at_position_bound"]
    )

    return {
        "fit_success": bool(fit_success),
        "x_2025": fit1["x"],
        "y_2025": fit1["y"],
        "x_2026": fit2["x"],
        "y_2026": fit2["y"],
        "dx_pixels": float(dx),
        "dy_pixels": float(dy),
        "displacement_pixels": displacement,
        "fit_rms_2025": fit1["fit_rms"],
        "fit_rms_2026": fit2["fit_rms"],
        "fit_contrast_2025": fit1["fit_contrast"],
        "fit_contrast_2026": fit2["fit_contrast"],
        "sigma_x_2025": fit1["sigma_x"],
        "sigma_y_2025": fit1["sigma_y"],
        "sigma_x_2026": fit2["sigma_x"],
        "sigma_y_2026": fit2["sigma_y"],
    }


if __name__ == "__main__":

    base_dir = Path(__file__).resolve().parent.parent

    raw_dir = base_dir / "data" / "raw"

    processed_dir = (
        base_dir / "data" / "processed" / "automatic"
    )

    print("\nSPHEREx Astrometric Centroid Analysis")
    print("====================================")

    pairs = Table.read(
        processed_dir / "D5_motion_pairs.csv",
        format="csv",
    )

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
        common_mask = data[
            "common_mask"
        ].astype(bool)

    valid_mask = (
        common_mask
        & np.isfinite(image1)
        & np.isfinite(image2)
    )

    results = []

    print("\nAnalyzing motion pairs...")

    for index, pair in enumerate(pairs, 1):

        result = analyze_motion_pair(
            image1,
            image2,
            valid_mask,
            pair,
        )

        result["pair_id"] = index
        result["negative_x"] = int(pair["negative_x"])
        result["negative_y"] = int(pair["negative_y"])
        result["positive_x"] = int(pair["positive_x"])
        result["positive_y"] = int(pair["positive_y"])

        results.append(result)

        print(
            f"Pair {index}: "
            f"fit_success={result['fit_success']} "
            f"displacement="
            f"{result['displacement_pixels']:.4f} px"
        )

    # Build a table with consistent columns even
    # when one or more source fits fail.
    columns = [
        "pair_id",
        "negative_x",
        "negative_y",
        "positive_x",
        "positive_y",
        "fit_success",
        "x_2025",
        "y_2025",
        "x_2026",
        "y_2026",
        "dx_pixels",
        "dy_pixels",
        "displacement_pixels",
        "fit_rms_2025",
        "fit_rms_2026",
        "fit_contrast_2025",
        "fit_contrast_2026",
        "sigma_x_2025",
        "sigma_y_2025",
        "sigma_x_2026",
        "sigma_y_2026",
    ]

    table = Table()

    for column in columns:
        if column == "fit_success":
            table[column] = [
                bool(row.get(column, False))
                for row in results
            ]
        elif column in (
            "pair_id",
            "negative_x",
            "negative_y",
            "positive_x",
            "positive_y",
        ):
            table[column] = [
                int(row.get(column, -1))
                for row in results
            ]
        else:
            table[column] = [
                float(row.get(column, np.nan))
                for row in results
            ]

    output_path = (
        processed_dir / "D5_astrometric_results.csv"
    )

    table.write(
        output_path,
        format="csv",
        overwrite=True,
    )

    print("\nSaved:", output_path)
    print("Astrometric analysis completed.")
