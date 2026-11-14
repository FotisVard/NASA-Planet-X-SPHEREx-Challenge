
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.stats import mad_std
from scipy.ndimage import (
    maximum_filter,
    minimum_filter,
    binary_erosion,
    median_filter,
)


def detect_candidates(
    difference,
    valid_mask,
    threshold=5.0,
    min_distance=5,
    border_distance=8,
):
    """
    Detect positive and negative local extrema.

    Scores are based on robust local residual scatter,
    not propagated photometric uncertainties.
    """

    valid = (
        valid_mask.astype(bool)
        & np.isfinite(difference)
    )

    if not np.any(valid):
        raise ValueError("No valid pixels.")

    # Avoid boundaries of invalid regions.
    safe_mask = binary_erosion(
        valid,
        structure=np.ones((3, 3), dtype=bool),
        iterations=border_distance,
        border_value=0,
    )

    # Estimate global center and robust scatter.
    values = difference[valid]
    global_median = float(np.median(values))
    global_sigma = float(mad_std(values))

    if not np.isfinite(global_sigma) or global_sigma <= 0:
        raise ValueError("Invalid robust scatter.")

    print("Global median:", global_median)
    print("Global robust scatter:", global_sigma)

    # Median-filtered background estimate.
    # Invalid pixels are filled only for filtering.
    filled = np.where(
        valid,
        difference,
        global_median,
    ).astype(np.float32)

    local_center = median_filter(
        filled,
        size=31,
        mode="nearest",
    )

    residual = filled - local_center

    # Local robust scatter from median absolute deviation.
    local_abs_dev = np.abs(residual)

    local_sigma = (
        1.4826
        * median_filter(
            local_abs_dev,
            size=31,
            mode="nearest",
        )
    )

    # Avoid unrealistically small noise estimates.
    local_sigma = np.maximum(
        local_sigma,
        0.5 * global_sigma,
    )

    score = np.full(
        difference.shape,
        np.nan,
        dtype=np.float32,
    )

    score[safe_mask] = (
        residual[safe_mask]
        / local_sigma[safe_mask]
    )

    # Find positive peaks.
    positive_map = np.where(
        safe_mask,
        score,
        -np.inf,
    )

    positive_peaks = (
        (positive_map >= threshold)
        & (
            positive_map
            == maximum_filter(
                positive_map,
                size=2 * min_distance + 1,
            )
        )
    )

    # Find negative peaks.
    negative_map = np.where(
        safe_mask,
        score,
        np.inf,
    )

    negative_peaks = (
        (negative_map <= -threshold)
        & (
            negative_map
            == minimum_filter(
                negative_map,
                size=2 * min_distance + 1,
            )
        )
    )

    candidates = []

    for polarity, peak_mask in [
        ("positive", positive_peaks),
        ("negative", negative_peaks),
    ]:

        ys, xs = np.where(peak_mask)

        for x, y in zip(xs, ys):
            candidates.append({
                "x": int(x),
                "y": int(y),
                "polarity": polarity,
                "residual_mjy_sr": float(residual[y, x]),
                "local_scatter_mjy_sr": float(
                    local_sigma[y, x]
                ),
                "score": float(score[y, x]),
                "abs_score": float(abs(score[y, x])),
            })

    candidates.sort(
        key=lambda row: row["abs_score"],
        reverse=True,
    )

    return candidates, score


def save_candidate_figure(
    difference,
    candidates,
    output_path,
    max_display=100,
):
    """
    Show the strongest candidate locations.
    """

    finite = difference[np.isfinite(difference)]

    limit = float(
        np.nanpercentile(np.abs(finite), 99)
    )

    if not np.isfinite(limit) or limit <= 0:
        limit = 1.0

    fig, ax = plt.subplots(
        figsize=(11, 10)
    )

    image = ax.imshow(
        difference,
        origin="lower",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )

    selected = candidates[:max_display]

    positive = [
        c for c in selected
        if c["polarity"] == "positive"
    ]

    negative = [
        c for c in selected
        if c["polarity"] == "negative"
    ]

    if positive:
        ax.scatter(
            [c["x"] for c in positive],
            [c["y"] for c in positive],
            facecolors="none",
            edgecolors="lime",
            s=75,
            linewidths=1.2,
            label="Positive residual",
        )

    if negative:
        ax.scatter(
            [c["x"] for c in negative],
            [c["y"] for c in negative],
            facecolors="none",
            edgecolors="cyan",
            s=75,
            linewidths=1.2,
            label="Negative residual",
        )

    ax.set_title(
        "SPHEREx-D5 Candidate Residuals"
    )

    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")

    if selected:
        ax.legend(loc="upper right")

    fig.colorbar(
        image,
        ax=ax,
        label="Residual brightness (MJy/sr)",
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)


if __name__ == "__main__":

    base_dir = Path(__file__).resolve().parent.parent

    output_dir = (
        base_dir / "data" / "processed" / "automatic"
    )

    input_path = (
        output_dir / "D5_psf_matching.npz"
    )

    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing input file: {input_path}"
        )

    print("\nSPHEREx Candidate Detection")
    print("==========================")

    print("\nLoading PSF-matched difference...")

    with np.load(input_path) as data:
        difference = data[
            "difference_after"
        ].astype(np.float32)

        valid_mask = data[
            "common_mask"
        ].astype(bool)

    print(
        "Valid pixels:",
        int(np.count_nonzero(valid_mask))
    )

    print("\nDetecting candidate residuals...")

    candidates, score = detect_candidates(
        difference,
        valid_mask,
        threshold=5.0,
        min_distance=5,
        border_distance=8,
    )

    print("\nCandidate Detection Results")
    print("---------------------------")
    print("Total candidates:", len(candidates))

    positive_count = sum(
        c["polarity"] == "positive"
        for c in candidates
    )

    negative_count = sum(
        c["polarity"] == "negative"
        for c in candidates
    )

    print("Positive candidates:", positive_count)
    print("Negative candidates:", negative_count)

    columns = [
        "x",
        "y",
        "polarity",
        "residual_mjy_sr",
        "local_scatter_mjy_sr",
        "score",
        "abs_score",
    ]

    table = Table(
        rows=[
            [candidate[col] for col in columns]
            for candidate in candidates
        ],
        names=columns,
    )

    csv_path = (
        output_dir / "D5_candidates.csv"
    )

    table.write(
        csv_path,
        format="csv",
        overwrite=True,
    )

    print("\nTop 10 candidates:")

    for candidate in candidates[:10]:
        print(
            f"({candidate['x']}, {candidate['y']})",
            candidate["polarity"],
            f"score={candidate['score']:.2f}",
        )

    figure_path = (
        output_dir / "D5_candidates.png"
    )

    save_candidate_figure(
        difference,
        candidates,
        figure_path,
    )

    print("\nSaved CSV:", csv_path)
    print("Saved figure:", figure_path)
    print("Candidate detection completed.")
