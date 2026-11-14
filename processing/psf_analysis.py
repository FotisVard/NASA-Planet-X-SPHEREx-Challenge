
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from scipy.optimize import curve_fit

from fits_loader import load_spherex_fits
from image_comparison import compare_images


def gaussian_2d(coords, amplitude, x0, y0, sigma, offset):
    x, y = coords

    model = offset + amplitude * np.exp(
        -((x - x0)**2 + (y - y0)**2) / (2 * sigma**2)
    )

    return model.ravel()


def measure_fwhm(image, x, y, radius=8):
    """
    Fit a circular Gaussian to a source cutout.
    Return approximate FWHM in pixels.
    """

    x = int(round(x))
    y = int(round(y))

    height, width = image.shape

    if (
        x - radius < 0
        or y - radius < 0
        or x + radius >= width
        or y + radius >= height
    ):
        return np.nan

    cutout = image[
        y-radius:y+radius+1,
        x-radius:x+radius+1
    ].astype(float)

    if not np.all(np.isfinite(cutout)):
        return np.nan

    yy, xx = np.indices(cutout.shape)

    background = np.median(cutout)
    amplitude = np.max(cutout) - background

    if amplitude <= 0:
        return np.nan

    initial = [
        amplitude,
        radius,
        radius,
        2.0,
        background
    ]

    try:
        parameters, _ = curve_fit(
            gaussian_2d,
            (xx, yy),
            cutout.ravel(),
            p0=initial,
            bounds=(
                [0, radius-3, radius-3, 0.5, -np.inf],
                [np.inf, radius+3, radius+3, 8.0, np.inf]
            ),
            maxfev=3000
        )

        sigma = parameters[3]

        fitted_x = parameters[1]
        fitted_y = parameters[2]

        if (
            abs(fitted_x - radius) > 2.0
            or abs(fitted_y - radius) > 2.0
        ):
            return np.nan

        if sigma <= 0.55 or sigma >= 7.9:
            return np.nan

        return 2.35482 * sigma

    except (RuntimeError, ValueError, TypeError):
        return np.nan


def get_centroid_columns(sources):
    """
    Identify the centroid column names returned
    by the installed photutils version.
    """

    print("\nAvailable source columns:")
    print(sources.colnames)

    candidates = [
        ("xcentroid", "ycentroid"),
        ("x_centroid", "y_centroid"),
        ("x_center", "y_center"),
    ]

    for x_name, y_name in candidates:
        if (
            x_name in sources.colnames
            and y_name in sources.colnames
        ):
            return x_name, y_name

    raise ValueError(
        "Cannot identify centroid columns. "
        f"Available columns: {sources.colnames}"
    )


def analyze_psf(result, max_sources=100):
    """
    Detect sources in image 1 and compare
    approximate FWHM in both aligned images.
    """

    image1 = result["image1"]
    image2 = result["aligned_image2"]
    valid = result["valid_mask"]

    valid = (
        valid
        & np.isfinite(image1)
        & np.isfinite(image2)
    )

    mean, median, std = sigma_clipped_stats(
        image1,
        sigma=3.0,
        mask=~valid
    )

    print("\nImage statistics:")
    print("Mean:", mean)
    print("Median:", median)
    print("Standard deviation:", std)

    detection_image = np.where(
        valid,
        image1 - median,
        0.0
    )

    finder = DAOStarFinder(
        fwhm=3.0,
        threshold=5.0 * std
    )

    sources = finder(detection_image)

    if sources is None or len(sources) == 0:
        raise ValueError(
            "No sources detected."
        )

    print("\nDetected sources:", len(sources))

    x_column, y_column = get_centroid_columns(
        sources
    )

    print("Using X column:", x_column)
    print("Using Y column:", y_column)

    if "flux" in sources.colnames:
        sources.sort("flux")
        sources.reverse()

    fwhm1 = []
    fwhm2 = []

    for source in sources[:max_sources]:

        x = float(source[x_column])
        y = float(source[y_column])

        width1 = measure_fwhm(
            image1, x, y
        )

        width2 = measure_fwhm(
            image2, x, y
        )

        if (
            np.isfinite(width1)
            and np.isfinite(width2)
        ):
            fwhm1.append(width1)
            fwhm2.append(width2)

    return (
        np.array(fwhm1),
        np.array(fwhm2)
    )


def save_psf_figure(fwhm1, fwhm2, output_path):

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    ax.scatter(
        fwhm1,
        fwhm2,
        alpha=0.6,
        label="Measured sources"
    )

    maximum = max(
        np.max(fwhm1),
        np.max(fwhm2)
    )

    ax.plot(
        [0, maximum],
        [0, maximum],
        "--",
        label="Equal FWHM"
    )

    ax.set_xlabel(
        "2025 FWHM (pixels)"
    )

    ax.set_ylabel(
        "2026 FWHM (pixels)"
    )

    ax.set_title(
        "SPHEREx PSF Comparison"
    )

    ax.legend()
    ax.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150
    )

    plt.close(fig)


if __name__ == "__main__":

    from urllib.parse import urlparse

    base_dir = Path(__file__).resolve().parent.parent

    raw_dir = base_dir / "data" / "raw"

    output_dir = (
        base_dir / "data" / "processed" / "automatic"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = Table.read(
        base_dir
        / "data"
        / "searches"
        / "spectral_ranked_pairs.csv",
        format="csv"
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

    print("\nSelected detector: SPHEREx-D5")
    print("Image 1:", filename1)
    print("Image 2:", filename2)

    print("\nLoading FITS observations...")

    data1 = load_spherex_fits(
        raw_dir / filename1
    )

    data2 = load_spherex_fits(
        raw_dir / filename2
    )

    print("\nComparing images...")

    result = compare_images(
        data1,
        data2
    )

    print("\nMeasuring source FWHM...")

    fwhm1, fwhm2 = analyze_psf(
        result,
        max_sources=100
    )

    if len(fwhm1) < 5:
        raise RuntimeError(
            "Insufficient valid PSF measurements."
        )

    median1 = float(np.median(fwhm1))
    median2 = float(np.median(fwhm2))

    print("\nPSF Analysis Results")
    print("--------------------")

    print("Matched sources:", len(fwhm1))

    print(
        "Median FWHM 2025:",
        median1,
        "pixels"
    )

    print(
        "Median FWHM 2026:",
        median2,
        "pixels"
    )

    print(
        "Median paired FWHM difference:",
        float(np.median(fwhm2 - fwhm1)),
        "pixels"
    )

    print(
        "FWHM 2025 range:",
        float(np.min(fwhm1)),
        "to",
        float(np.max(fwhm1))
    )

    print(
        "FWHM 2026 range:",
        float(np.min(fwhm2)),
        "to",
        float(np.max(fwhm2))
    )

    np.savez_compressed(
        output_dir / "D5_psf_measurements.npz",
        fwhm_2025=fwhm1,
        fwhm_2026=fwhm2,
        median_fwhm_2025=median1,
        median_fwhm_2026=median2
    )

    output_path = (
        output_dir / "D5_psf_analysis.png"
    )

    save_psf_figure(
        fwhm1,
        fwhm2,
        output_path
    )

    print("\nSaved:", output_path)
    print("PSF analysis completed.")
