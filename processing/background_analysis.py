
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.stats import SigmaClip

from photutils.background import (
    Background2D,
    MedianBackground,
)

from fits_loader import load_spherex_fits
from image_comparison import compare_images


def estimate_local_background(difference, valid_mask):
    """
    Estimate a spatially varying background
    from the difference image.
    """

    sigma_clip = SigmaClip(
        sigma=3.0,
        maxiters=5
    )

    estimator = MedianBackground()

    # Background2D excludes masked pixels.
    mask = (
        ~valid_mask
        | ~np.isfinite(difference)
    )

    background = Background2D(
        difference,
        box_size=(128, 128),
        filter_size=(3, 3),
        mask=mask,
        sigma_clip=sigma_clip,
        bkg_estimator=estimator,
        exclude_percentile=50.0
    )

    return background.background


def correct_background(result):
    """
    Subtract the estimated local background
    from the difference map.
    """

    difference = result["difference"]
    valid_mask = result["valid_mask"]

    background_map = estimate_local_background(
        difference,
        valid_mask
    )

    corrected_difference = np.full(
        difference.shape,
        np.nan,
        dtype=np.float32
    )

    corrected_difference[valid_mask] = (
        difference[valid_mask]
        - background_map[valid_mask]
    )

    return {
        "background_map": background_map,
        "corrected_difference": corrected_difference,
        "valid_mask": valid_mask
    }


def save_background_figure(
    difference,
    background_map,
    corrected_difference,
    output_path
):

    fig, axes = plt.subplots(
        1, 3,
        figsize=(18, 6)
    )

    images = [
        difference,
        background_map,
        corrected_difference
    ]

    titles = [
        "Original Difference",
        "Estimated Local Background",
        "Background-Corrected Difference"
    ]

    for ax, image, title in zip(
        axes, images, titles
    ):
        im = ax.imshow(
            image,
            origin="lower",
            cmap="coolwarm",
            vmin=-0.3,
            vmax=0.3
        )

        ax.set_title(title)
        ax.set_xlabel("X pixel")
        ax.set_ylabel("Y pixel")

    fig.colorbar(
        im,
        ax=axes,
        label="MJy/sr",
        shrink=0.8
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

    filename1 = str(pair["url_1"]).split("/")[-1]
    filename2 = str(pair["url_2"]).split("/")[-1]

    print("\nSelected detector: SPHEREx-D5")
    print("Image 1:", filename1)
    print("Image 2:", filename2)

    print("\nLoading SPHEREx observations...")

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

    print("\nEstimating local background...")

    corrected = correct_background(result)

    valid = corrected["valid_mask"]

    original = result["difference"][valid]

    corrected_values = (
        corrected["corrected_difference"][valid]
    )

    def robust_scatter(values):
        median = np.median(values)
        return 1.4826 * np.median(
            np.abs(values - median)
        )

    print("\nBackground Analysis Results")
    print("---------------------------")

    print("Valid pixels:", int(np.sum(valid)))

    print(
        "Original median:",
        float(np.median(original)),
        "MJy/sr"
    )

    print(
        "Corrected median:",
        float(np.median(corrected_values)),
        "MJy/sr"
    )

    print(
        "Original robust scatter:",
        float(robust_scatter(original)),
        "MJy/sr"
    )

    print(
        "Corrected robust scatter:",
        float(robust_scatter(corrected_values)),
        "MJy/sr"
    )

    background_values = (
        corrected["background_map"][valid]
    )

    print(
        "Background minimum:",
        float(np.min(background_values)),
        "MJy/sr"
    )

    print(
        "Background maximum:",
        float(np.max(background_values)),
        "MJy/sr"
    )

    output_path = (
        output_dir / "D5_background_corrected.png"
    )

    save_background_figure(
        result["difference"],
        corrected["background_map"],
        corrected["corrected_difference"],
        output_path
    )

    np.savez_compressed(
        output_dir / "D5_background_corrected.npz",
        corrected_difference=(
            corrected["corrected_difference"]
        ),
        background_map=corrected["background_map"],
        valid_mask=valid
    )

    print("\nSaved figure:", output_path)
    print("Saved corrected data: D5_background_corrected.npz")
    print("Background analysis completed.")
