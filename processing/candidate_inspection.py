
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table

from fits_loader import load_spherex_fits
from image_comparison import compare_images


def extract_cutout(image, x, y, radius=20):
    """
    Extract a square cutout centered on a candidate.
    """

    x = int(round(x))
    y = int(round(y))

    height, width = image.shape

    x_min = max(0, x - radius)
    x_max = min(width, x + radius + 1)

    y_min = max(0, y - radius)
    y_max = min(height, y + radius + 1)

    return image[
        y_min:y_max,
        x_min:x_max
    ]


def robust_limits(image1, image2):
    """
    Use common display limits for the two epochs.
    """

    a = image1[np.isfinite(image1)]
    b = image2[np.isfinite(image2)]

    if len(a) == 0 or len(b) == 0:
        return 0.0, 1.0

    combined = np.concatenate([a, b])

    low, high = np.percentile(
        combined,
        [2, 98]
    )

    if high <= low:
        high = low + 1.0

    return float(low), float(high)


def inspect_candidate(
    image1,
    image2,
    difference,
    candidate,
    output_path,
    radius=20
):
    """
    Generate a three-panel inspection figure.
    """

    x = int(candidate["x"])
    y = int(candidate["y"])

    cut1 = extract_cutout(
        image1, x, y, radius
    )

    cut2 = extract_cutout(
        image2, x, y, radius
    )

    cut_diff = extract_cutout(
        difference, x, y, radius
    )

    vmin, vmax = robust_limits(cut1, cut2)

    finite_diff = cut_diff[
        np.isfinite(cut_diff)
    ]

    if len(finite_diff):
        limit = float(
            np.percentile(
                np.abs(finite_diff),
                99
            )
        )
    else:
        limit = 1.0

    if limit <= 0:
        limit = 1.0

    fig, axes = plt.subplots(
        1, 3, figsize=(13, 4.5)
    )

    axes[0].imshow(
        cut1,
        origin="lower",
        cmap="gray",
        vmin=vmin,
        vmax=vmax
    )

    axes[0].set_title("2025")

    axes[1].imshow(
        cut2,
        origin="lower",
        cmap="gray",
        vmin=vmin,
        vmax=vmax
    )

    axes[1].set_title("2026 (Aligned)")

    im = axes[2].imshow(
        cut_diff,
        origin="lower",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit
    )

    axes[2].set_title(
        "PSF-Matched Difference"
    )

    for ax in axes:
        ax.set_xlabel("X offset (pixels)")
        ax.set_ylabel("Y offset (pixels)")

        ax.axvline(
            radius,
            color="yellow",
            linestyle="--",
            alpha=0.6
        )

        ax.axhline(
            radius,
            color="yellow",
            linestyle="--",
            alpha=0.6
        )

    fig.colorbar(
        im,
        ax=axes[2],
        shrink=0.75,
        label="MJy/sr"
    )

    fig.suptitle(
        f"Candidate ({x}, {y}) | "
        f"{candidate['polarity']} | "
        f"score={float(candidate['score']):.2f}"
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

    processed_dir = (
        base_dir / "data" / "processed" / "automatic"
    )

    inspection_dir = (
        processed_dir / "candidate_inspection"
    )

    inspection_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nSPHEREx Candidate Inspection")
    print("===========================")

    candidates = Table.read(
        processed_dir / "D5_candidates_validated.csv",
        format="csv"
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
            "No compatible D5 pair found."
        )

    pair = matches[0]

    filename1 = Path(
        urlparse(str(pair["url_1"])).path
    ).name

    filename2 = Path(
        urlparse(str(pair["url_2"])).path
    ).name

    print("\nLoading observations...")

    data1 = load_spherex_fits(
        raw_dir / filename1
    )

    data2 = load_spherex_fits(
        raw_dir / filename2
    )

    print("\nAligning images...")

    comparison = compare_images(
        data1,
        data2
    )

    image1 = comparison["image1"]
    image2 = comparison["aligned_image2"]

    with np.load(
        processed_dir / "D5_psf_matching.npz"
    ) as data:
        difference = data[
            "difference_after"
        ].astype(np.float32)

    # Inspect candidates without nearby
    # opposite-sign peaks first.
    candidates.sort("abs_score")
    candidates.reverse()

    preferred = candidates[
        candidates["review_status"] == "needs_review"
    ]

    print(
        "\nCandidates for first inspection:",
        len(preferred)
    )

    for index, candidate in enumerate(
        preferred,
        start=1
    ):

        x = int(candidate["x"])
        y = int(candidate["y"])

        output_path = (
            inspection_dir
            / f"candidate_{index:02d}_x{x}_y{y}.png"
        )

        print(
            f"Inspecting {index}/{len(preferred)}: "
            f"({x}, {y})"
        )

        inspect_candidate(
            image1,
            image2,
            difference,
            candidate,
            output_path,
            radius=20
        )

    print("\nSaved inspection images to:")
    print(inspection_dir)

    print("\nCandidate inspection completed.")
