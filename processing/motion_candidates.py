
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from scipy.spatial import cKDTree


def find_motion_pairs(
    candidates,
    max_distance=8.0,
    min_distance=0.5,
):
    """
    Match negative residual peaks to positive peaks.

    A negative peak represents a possible old position.
    A positive peak represents a possible new position.

    All combinations within the distance limits are
    retained. These are hypotheses, not confirmed motion.
    """

    positive = candidates[
        candidates["polarity"] == "positive"
    ]

    negative = candidates[
        candidates["polarity"] == "negative"
    ]

    if len(positive) == 0 or len(negative) == 0:
        return []

    positive_coords = np.column_stack([
        positive["x"],
        positive["y"],
    ]).astype(float)

    negative_coords = np.column_stack([
        negative["x"],
        negative["y"],
    ]).astype(float)

    tree = cKDTree(positive_coords)

    pairs = []

    for negative_index, negative_candidate in enumerate(
        negative
    ):
        old_position = negative_coords[negative_index]

        nearby_indices = tree.query_ball_point(
            old_position,
            r=max_distance,
        )

        for positive_index in nearby_indices:
            positive_candidate = positive[positive_index]

            new_position = positive_coords[positive_index]

            dx = float(new_position[0] - old_position[0])
            dy = float(new_position[1] - old_position[1])

            distance = float(np.hypot(dx, dy))

            if distance < min_distance:
                continue

            angle = float(
                np.degrees(np.arctan2(dy, dx))
            )

            negative_strength = abs(
                float(negative_candidate["score"])
            )

            positive_strength = abs(
                float(positive_candidate["score"])
            )

            # Symmetry of preliminary peak scores.
            # This is not a calibrated probability.
            strength_balance = (
                min(negative_strength, positive_strength)
                / max(negative_strength, positive_strength)
            )

            pairs.append({
                "negative_x": int(old_position[0]),
                "negative_y": int(old_position[1]),
                "positive_x": int(new_position[0]),
                "positive_y": int(new_position[1]),
                "dx_pixels": dx,
                "dy_pixels": dy,
                "distance_pixels": distance,
                "angle_degrees": angle,
                "negative_score": float(
                    negative_candidate["score"]
                ),
                "positive_score": float(
                    positive_candidate["score"]
                ),
                "strength_balance": float(
                    strength_balance
                ),
            })

    # Balanced peak strengths are useful for inspection,
    # but not sufficient evidence of motion.
    pairs.sort(
        key=lambda pair: (
            -pair["strength_balance"],
            pair["distance_pixels"],
        )
    )

    return pairs


def plot_motion_pairs(
    difference,
    pairs,
    output_path,
    max_pairs=20,
):
    finite = difference[np.isfinite(difference)]

    if len(finite) == 0:
        raise ValueError("No finite difference pixels.")

    limit = float(
        np.percentile(np.abs(finite), 99)
    )

    if limit <= 0:
        limit = 1.0

    fig, ax = plt.subplots(figsize=(11, 10))

    image = ax.imshow(
        difference,
        origin="lower",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
    )

    for index, pair in enumerate(pairs[:max_pairs], 1):
        x1 = pair["negative_x"]
        y1 = pair["negative_y"]

        x2 = pair["positive_x"]
        y2 = pair["positive_y"]

        ax.plot(
            x1,
            y1,
            "o",
            markerfacecolor="none",
            markeredgecolor="cyan",
            markersize=8,
        )

        ax.plot(
            x2,
            y2,
            "o",
            markerfacecolor="none",
            markeredgecolor="lime",
            markersize=8,
        )

        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops={
                "arrowstyle": "->",
                "color": "yellow",
                "lw": 1.5,
            },
        )

        ax.text(
            x2 + 6,
            y2 + 6,
            str(index),
            color="yellow",
            fontsize=8,
        )

    ax.set_title(
        "SPHEREx-D5: Preliminary Motion Pair Hypotheses"
    )

    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")

    fig.colorbar(
        image,
        ax=ax,
        label="Difference brightness (MJy/sr)",
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

    candidates_path = (
        output_dir / "D5_candidate_quality.csv"
    )

    difference_path = (
        output_dir / "D5_psf_matching.npz"
    )

    print("\nSPHEREx Motion Candidate Analysis")
    print("=================================")

    candidates = Table.read(
        candidates_path,
        format="csv",
    )

    print("Loaded candidates:", len(candidates))

    pairs = find_motion_pairs(
        candidates,
        max_distance=8.0,
        min_distance=0.5,
    )

    print("\nMotion Pair Results")
    print("-------------------")
    print("Total pair hypotheses:", len(pairs))

    columns = [
        "negative_x",
        "negative_y",
        "positive_x",
        "positive_y",
        "dx_pixels",
        "dy_pixels",
        "distance_pixels",
        "angle_degrees",
        "negative_score",
        "positive_score",
        "strength_balance",
    ]

    table = Table(
        rows=[
            [pair[column] for column in columns]
            for pair in pairs
        ],
        names=columns,
    )

    csv_path = (
        output_dir / "D5_motion_pairs.csv"
    )

    table.write(
        csv_path,
        format="csv",
        overwrite=True,
    )

    print("\nTop 10 pair hypotheses:")

    for index, pair in enumerate(pairs[:10], 1):
        print(
            f"{index}. "
            f"({pair['negative_x']}, {pair['negative_y']})"
            f" -> "
            f"({pair['positive_x']}, {pair['positive_y']})"
            f" | distance={pair['distance_pixels']:.2f}px"
            f" | balance={pair['strength_balance']:.3f}"
        )

    with np.load(difference_path) as data:
        difference = data[
            "difference_after"
        ].astype(np.float32)

    figure_path = (
        output_dir / "D5_motion_pairs.png"
    )

    plot_motion_pairs(
        difference,
        pairs,
        figure_path,
    )

    print("\nSaved CSV:", csv_path)
    print("Saved figure:", figure_path)

    print("\nMotion candidate analysis completed.")
