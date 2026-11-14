
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from scipy.stats import theilslopes

from fits_loader import load_spherex_fits
from image_comparison import compare_images


def analyze_photometry(result, sample_size=3000):
    """
    Estimate a robust linear relationship between
    two aligned SPHEREx images.

    This is a diagnostic pixel-level fit, not
    a calibrated stellar photometric solution.
    """

    image1 = result["image1"]
    image2 = result["aligned_image2"]
    valid = result["valid_mask"]

    selected = (
        valid
        & np.isfinite(image1)
        & np.isfinite(image2)
        & (image1 > 0)
        & (image2 > 0)
    )

    y_pixels, x_pixels = np.where(selected)

    if len(x_pixels) < 100:
        raise ValueError(
            "Not enough valid pixels for analysis."
        )

    rng = np.random.default_rng(42)

    indices = rng.choice(
        len(x_pixels),
        size=min(sample_size, len(x_pixels)),
        replace=False
    )

    x = image1[
        y_pixels[indices],
        x_pixels[indices]
    ].astype(float)

    y = image2[
        y_pixels[indices],
        x_pixels[indices]
    ].astype(float)

    # Remove extreme brightness values from the
    # diagnostic fit to limit bright-source influence.
    x_limit = np.percentile(x, 99)
    y_limit = np.percentile(y, 99)

    keep = (
        (x <= x_limit)
        & (y <= y_limit)
    )

    x = x[keep]
    y = y[keep]

    if len(x) < 100:
        raise ValueError(
            "Too few pixels after filtering."
        )

    slope, intercept, _, _ = theilslopes(
        y, x
    )

    predicted = slope * x + intercept
    residuals = y - predicted

    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "residual_median": float(
            np.median(residuals)
        ),
        "residual_mad": float(
            1.4826 * np.median(
                np.abs(residuals - np.median(residuals))
            )
        ),
        "x": x,
        "y": y,
    }


def save_photometric_figure(analysis, output_path):

    x = analysis["x"]
    y = analysis["y"]

    slope = analysis["slope"]
    intercept = analysis["intercept"]

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    ax.scatter(
        x, y,
        s=4,
        alpha=0.2
    )

    x_line = np.linspace(
        np.min(x),
        np.max(x),
        100
    )

    ax.plot(
        x_line,
        slope * x_line + intercept,
        linewidth=2,
        label="Robust fit"
    )

    ax.plot(
        x_line,
        x_line,
        linestyle="--",
        label="Ideal 1:1 relation"
    )

    ax.set_xlabel(
        "Image 1 brightness (MJy/sr)"
    )

    ax.set_ylabel(
        "Image 2 brightness (MJy/sr)"
    )

    ax.set_title(
        "SPHEREx Photometric Comparison"
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

    base_dir = Path(__file__).resolve().parent.parent

    raw_dir = base_dir / "data" / "raw"
    output_dir = base_dir / "data" / "processed"

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    pairs = Table.read(
        base_dir / "data" / "candidate_pairs.csv",
        format="csv"
    )

    pair = pairs[0]

    filename1 = str(pair["url_1"]).split("/")[-1]
    filename2 = str(pair["url_2"]).split("/")[-1]

    print("\nLoading SPHEREx observations...")

    data1 = load_spherex_fits(
        raw_dir / filename1
    )

    data2 = load_spherex_fits(
        raw_dir / filename2
    )

    print("Image 1:", filename1)
    print("Image 2:", filename2)

    print("\nRunning photometric analysis...")

    result = compare_images(
        data1, data2
    )

    analysis = analyze_photometry(result)

    print("\nPhotometric Analysis Results")
    print("----------------------------")

    print(
        "Photometric scale (a):",
        analysis["slope"]
    )

    print(
        "Brightness offset (b):",
        analysis["intercept"],
        "MJy/sr"
    )

    print(
        "Residual median:",
        analysis["residual_median"],
        "MJy/sr"
    )

    print(
        "Residual robust scatter:",
        analysis["residual_mad"],
        "MJy/sr"
    )

    output_path = (
        output_dir / "photometric_analysis.png"
    )

    save_photometric_figure(
        analysis, output_path
    )

    print("\nSaved figure:", output_path)
    print("Photometric analysis completed.")
