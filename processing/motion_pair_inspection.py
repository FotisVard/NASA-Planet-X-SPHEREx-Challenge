
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table

from fits_loader import load_spherex_fits
from image_comparison import compare_images


def get_cutout(image, center_x, center_y, radius=25):
    """Return cutout and global pixel boundaries."""

    height, width = image.shape

    x0 = max(0, int(center_x) - radius)
    x1 = min(width, int(center_x) + radius + 1)

    y0 = max(0, int(center_y) - radius)
    y1 = min(height, int(center_y) + radius + 1)

    return image[y0:y1, x0:x1], x0, y0


def display_limits(image1, image2):
    """Shared intensity limits for both epochs."""

    values = np.concatenate([
        image1[np.isfinite(image1)],
        image2[np.isfinite(image2)],
    ])

    if len(values) == 0:
        return 0.0, 1.0

    low, high = np.percentile(values, [2, 98])

    if high <= low:
        high = low + 1.0

    return float(low), float(high)


def inspect_motion_pair(
    image1,
    image2,
    difference,
    pair,
    output_path,
    radius=25,
):
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

    cut1, x0, y0 = get_cutout(
        image1, center_x, center_y, radius
    )

    cut2, _, _ = get_cutout(
        image2, center_x, center_y, radius
    )

    cut_diff, _, _ = get_cutout(
        difference, center_x, center_y, radius
    )

    vmin, vmax = display_limits(cut1, cut2)

    finite_diff = cut_diff[np.isfinite(cut_diff)]

    if len(finite_diff):
        diff_limit = float(
            np.percentile(
                np.abs(finite_diff),
                99
            )
        )
    else:
        diff_limit = 1.0

    if diff_limit <= 0:
        diff_limit = 1.0

    old_x = negative_x - x0
    old_y = negative_y - y0

    new_x = positive_x - x0
    new_y = positive_y - y0

    fig, axes = plt.subplots(
        1, 3,
        figsize=(15, 5),
        constrained_layout=True,
    )

    axes[0].imshow(
        cut1,
        origin="lower",
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )

    axes[0].set_title("2025")

    axes[1].imshow(
        cut2,
        origin="lower",
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )

    axes[1].set_title("2026 (Aligned)")

    image = axes[2].imshow(
        cut_diff,
        origin="lower",
        cmap="RdBu_r",
        vmin=-diff_limit,
        vmax=diff_limit,
    )

    axes[2].set_title("PSF-Matched Difference")

    for ax in axes:
        ax.scatter(
            old_x,
            old_y,
            s=120,
            facecolors="none",
            edgecolors="cyan",
            linewidths=1.8,
            label="Negative peak",
        )

        ax.scatter(
            new_x,
            new_y,
            s=120,
            facecolors="none",
            edgecolors="red",
            linewidths=1.8,
            label="Positive peak",
        )

        ax.set_xlabel("X offset (pixels)")
        ax.set_ylabel("Y offset (pixels)")

    axes[2].annotate(
        "",
        xy=(new_x, new_y),
        xytext=(old_x, old_y),
        arrowprops={
            "arrowstyle": "->",
            "color": "yellow",
            "lw": 1.5,
        },
    )

    axes[2].legend(
        loc="upper right",
        fontsize=8,
    )

    fig.colorbar(
        image,
        ax=axes[2],
        shrink=0.8,
        label="MJy/sr",
    )

    fig.suptitle(
        f"Motion Pair: "
        f"({negative_x}, {negative_y}) -> "
        f"({positive_x}, {positive_y}) | "
        f"Distance: {float(pair['distance_pixels']):.2f} px"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


if __name__ == "__main__":

    base_dir = Path(__file__).resolve().parent.parent

    raw_dir = base_dir / "data" / "raw"

    processed_dir = (
        base_dir / "data" / "processed" / "automatic"
    )

    inspection_dir = (
        processed_dir / "motion_pair_inspection"
    )

    inspection_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("\nSPHEREx Motion Pair Inspection")
    print("=============================")

    pairs = Table.read(
        processed_dir / "D5_motion_pairs.csv",
        format="csv",
    )

    # Explicitly use the same D5 FITS pair
    # as in our previous processing steps.
    file1 = raw_dir / (
        "level2_2025W19_1B_0181_3D5_"
        "spx_l2b-v27-2026-197.fits"
    )

    file2 = raw_dir / (
        "level2_2026W20_1A_1166_1D5_"
        "spx_l2b-v27-2026-246.fits"
    )

    for path in (file1, file2):
        if not path.exists():
            raise FileNotFoundError(path)

    print("\nLoading FITS observations...")

    data1 = load_spherex_fits(file1)
    data2 = load_spherex_fits(file2)

    print("\nAligning observations...")

    comparison = compare_images(data1, data2)

    image1 = comparison["image1"]
    image2 = comparison["aligned_image2"]

    with np.load(
        processed_dir / "D5_psf_matching.npz"
    ) as data:
        difference = data[
            "difference_after"
        ].astype(np.float32)

        common_mask = data[
            "common_mask"
        ].astype(bool)

    if not (
        image1.shape == image2.shape
        == difference.shape == common_mask.shape
    ):
        raise ValueError(
            "Input image dimensions do not match."
        )

    difference[~common_mask] = np.nan

    print("\nTotal motion pairs:", len(pairs))

    for index, pair in enumerate(pairs, start=1):

        output_path = (
            inspection_dir
            / f"motion_pair_{index:02d}.png"
        )

        print(
            f"Inspecting pair {index}/{len(pairs)}: "
            f"({pair['negative_x']}, {pair['negative_y']}) "
            f"-> "
            f"({pair['positive_x']}, {pair['positive_y']})"
        )

        inspect_motion_pair(
            image1,
            image2,
            difference,
            pair,
            output_path,
            radius=25,
        )

    print("\nSaved inspection figures to:")
    print(inspection_dir)

    print("\nMotion pair inspection completed.")
