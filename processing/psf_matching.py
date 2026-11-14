
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from scipy.ndimage import gaussian_filter

from fits_loader import load_spherex_fits
from image_comparison import compare_images
from background_analysis import estimate_local_background


GAUSSIAN_FACTOR = 2.35482


def calculate_kernel_sigma(fwhm_source, fwhm_target):
    """
    Calculate Gaussian kernel sigma needed to broaden
    a sharper image toward a broader target PSF.
    """

    if fwhm_target <= fwhm_source:
        raise ValueError(
            "Target FWHM must exceed source FWHM."
        )

    sigma = np.sqrt(
        fwhm_target**2 - fwhm_source**2
    ) / GAUSSIAN_FACTOR

    return float(sigma)


def calculate_dynamic_psf_kernel(measurements_path):
    """
    Load measured FWHM values and determine
    which image requires Gaussian smoothing.
    """

    if not measurements_path.exists():
        raise FileNotFoundError(
            f"PSF measurements not found: {measurements_path}\n"
            "Run python processing/psf_analysis.py first."
        )

    with np.load(measurements_path) as data:
        fwhm1 = float(data["median_fwhm_2025"])
        fwhm2 = float(data["median_fwhm_2026"])

    if not (
        np.isfinite(fwhm1)
        and np.isfinite(fwhm2)
        and fwhm1 > 0
        and fwhm2 > 0
    ):
        raise ValueError("Invalid FWHM measurements.")

    print("\nMeasured PSF widths:")
    print("FWHM 2025:", fwhm1, "pixels")
    print("FWHM 2026:", fwhm2, "pixels")

    if np.isclose(fwhm1, fwhm2, atol=0.01, rtol=0):
        return 0.0, "none"

    if fwhm1 < fwhm2:
        sigma = calculate_kernel_sigma(fwhm1, fwhm2)
        return sigma, "image1"

    sigma = calculate_kernel_sigma(fwhm2, fwhm1)
    return sigma, "image2"


def masked_gaussian_filter(image, valid_mask, sigma):
    """
    Apply Gaussian convolution while excluding
    invalid pixels using normalized weights.
    """

    valid = (
        valid_mask
        & np.isfinite(image)
    )

    if sigma <= 0:
        output = image.astype(np.float32).copy()
        output[~valid] = np.nan
        return output

    values = np.where(
        valid,
        image,
        0.0
    ).astype(np.float32)

    weights = valid.astype(np.float32)

    filtered_values = gaussian_filter(
        values,
        sigma=sigma,
        mode="constant",
        cval=0.0
    )

    filtered_weights = gaussian_filter(
        weights,
        sigma=sigma,
        mode="constant",
        cval=0.0
    )

    output = np.full(
        image.shape,
        np.nan,
        dtype=np.float32
    )

    np.divide(
        filtered_values,
        filtered_weights,
        out=output,
        where=filtered_weights > 0.99
    )

    return output


def robust_scatter(values):
    """
    Calculate robust scatter using
    median absolute deviation (MAD).
    """

    values = values[np.isfinite(values)]

    if len(values) == 0:
        return np.nan

    median = np.median(values)

    return float(
        1.4826 * np.median(
            np.abs(values - median)
        )
    )


def analyze_psf_matching(
    result,
    kernel_sigma,
    smooth_image
):
    """
    Perform PSF matching and compare
    background-corrected difference images.
    """

    image1 = result["image1"]
    image2 = result["aligned_image2"]

    valid = (
        result["valid_mask"]
        & np.isfinite(image1)
        & np.isfinite(image2)
    )

    print("\nApplying PSF matching...")

    matched_image1 = image1
    matched_image2 = image2

    if smooth_image == "image1":

        print("Smoothing image 1 (2025)...")

        matched_image1 = masked_gaussian_filter(
            image1,
            valid,
            kernel_sigma
        )

    elif smooth_image == "image2":

        print("Smoothing image 2 (2026)...")

        matched_image2 = masked_gaussian_filter(
            image2,
            valid,
            kernel_sigma
        )

    elif smooth_image == "none":
        print("No smoothing required.")

    else:
        raise ValueError(
            f"Unknown smoothing direction: {smooth_image}"
        )

    common_mask = (
        valid
        & np.isfinite(matched_image1)
        & np.isfinite(matched_image2)
    )

    common_pixels = int(
        np.count_nonzero(common_mask)
    )

    if common_pixels == 0:
        raise RuntimeError(
            "No valid pixels after PSF matching."
        )

    print("Common valid pixels:", common_pixels)

    print("\nCalculating difference images...")

    difference_before = (
        image2 - image1
    ).astype(np.float32)

    difference_after = (
        matched_image2 - matched_image1
    ).astype(np.float32)

    difference_before[~common_mask] = np.nan
    difference_after[~common_mask] = np.nan

    print("\nEstimating local backgrounds...")

    background_before = estimate_local_background(
        difference_before,
        common_mask
    )

    background_after = estimate_local_background(
        difference_after,
        common_mask
    )

    corrected_before = (
        difference_before - background_before
    ).astype(np.float32)

    corrected_after = (
        difference_after - background_after
    ).astype(np.float32)

    corrected_before[~common_mask] = np.nan
    corrected_after[~common_mask] = np.nan

    print("\nCalculating robust scatter...")

    before_values = corrected_before[common_mask]
    after_values = corrected_after[common_mask]

    before_scatter = robust_scatter(before_values)
    after_scatter = robust_scatter(after_values)

    if (
        np.isfinite(before_scatter)
        and before_scatter > 0
    ):
        improvement = (
            100.0
            * (before_scatter - after_scatter)
            / before_scatter
        )
    else:
        improvement = np.nan

    return {
        "matched_image1": matched_image1,
        "matched_image2": matched_image2,
        "common_mask": common_mask,
        "difference_before": corrected_before,
        "difference_after": corrected_after,
        "scatter_before": before_scatter,
        "scatter_after": after_scatter,
        "improvement_percent": improvement
    }


def save_matching_figure(results, output_path):
    """
    Save a side-by-side comparison using
    the same intensity scale for both images.
    """

    before = results["difference_before"]
    after = results["difference_after"]
    mask = results["common_mask"]

    sample_before = np.abs(before[mask])
    sample_after = np.abs(after[mask])

    limit = max(
        np.nanpercentile(sample_before, 99),
        np.nanpercentile(sample_after, 99)
    )

    if not np.isfinite(limit) or limit <= 0:
        limit = 1.0

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 7)
    )

    axes[0].imshow(
        before,
        origin="lower",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit
    )

    axes[0].set_title(
        "Before PSF Matching"
    )

    im = axes[1].imshow(
        after,
        origin="lower",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit
    )

    axes[1].set_title(
        "After PSF Matching"
    )

    for ax in axes:
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")

    fig.colorbar(
        im,
        ax=axes,
        label="Residual brightness (MJy/sr)",
        shrink=0.75
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)


if __name__ == "__main__":

    base_dir = Path(__file__).resolve().parent.parent

    raw_dir = base_dir / "data" / "raw"

    output_dir = (
        base_dir / "data" / "processed" / "automatic"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nSPHEREx Dynamic PSF Matching")
    print("===========================")

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
            "No spectrally compatible D5 pair found."
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

    print("\nAligning and comparing images...")

    comparison = compare_images(
        data1,
        data2
    )

    measurements_path = (
        output_dir / "D5_psf_measurements.npz"
    )

    kernel_sigma, smooth_image = (
        calculate_dynamic_psf_kernel(
            measurements_path
        )
    )

    print("\nGaussian kernel sigma:", kernel_sigma)
    print("Image to smooth:", smooth_image)

    results = analyze_psf_matching(
        comparison,
        kernel_sigma,
        smooth_image
    )

    print("\nPSF Matching Results")
    print("--------------------")

    print(
        "Common valid pixels:",
        int(np.count_nonzero(
            results["common_mask"]
        ))
    )

    print(
        "Robust scatter before:",
        results["scatter_before"],
        "MJy/sr"
    )

    print(
        "Robust scatter after:",
        results["scatter_after"],
        "MJy/sr"
    )

    print(
        "Scatter improvement:",
        results["improvement_percent"],
        "%"
    )

    output_path = (
        output_dir / "D5_psf_matching.png"
    )

    print("\nSaving comparison figure...")

    save_matching_figure(
        results,
        output_path
    )

    np.savez_compressed(
        output_dir / "D5_psf_matching.npz",
        difference_before=results["difference_before"],
        difference_after=results["difference_after"],
        common_mask=results["common_mask"],
        kernel_sigma=np.float64(kernel_sigma),
        smooth_image=np.array(smooth_image)
    )

    print("\nSaved figure:", output_path)
    print("Saved data: D5_psf_matching.npz")
    print("\nDynamic PSF matching completed.")
