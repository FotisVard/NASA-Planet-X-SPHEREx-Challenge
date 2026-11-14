
from pathlib import Path

import numpy as np
from astropy.table import Table
from scipy.spatial import cKDTree


def validate_candidates(candidates, radius=3.0):
    """
    Flag candidates with an opposite-sign
    peak within a specified pixel radius.

    This is a diagnostic flag, not proof
    that a candidate is an artifact.
    """

    positive = candidates[
        candidates["polarity"] == "positive"
    ]

    negative = candidates[
        candidates["polarity"] == "negative"
    ]

    flags = np.zeros(
        len(candidates),
        dtype=bool
    )

    nearest_distances = np.full(
        len(candidates),
        np.nan,
        dtype=float
    )

    if len(positive) == 0 or len(negative) == 0:
        return flags, nearest_distances

    positive_coords = np.column_stack([
        positive["x"],
        positive["y"]
    ]).astype(float)

    negative_coords = np.column_stack([
        negative["x"],
        negative["y"]
    ]).astype(float)

    positive_tree = cKDTree(positive_coords)
    negative_tree = cKDTree(negative_coords)

    positive_indices = np.where(
        candidates["polarity"] == "positive"
    )[0]

    negative_indices = np.where(
        candidates["polarity"] == "negative"
    )[0]

    distances, _ = negative_tree.query(
        positive_coords
    )

    nearest_distances[positive_indices] = distances
    flags[positive_indices] = distances <= radius

    distances, _ = positive_tree.query(
        negative_coords
    )

    nearest_distances[negative_indices] = distances
    flags[negative_indices] = distances <= radius

    return flags, nearest_distances


if __name__ == "__main__":

    base_dir = Path(__file__).resolve().parent.parent

    output_dir = (
        base_dir / "data" / "processed" / "automatic"
    )

    input_path = output_dir / "D5_candidates.csv"

    candidates = Table.read(
        input_path,
        format="csv"
    )

    flags, distances = validate_candidates(
        candidates,
        radius=3.0
    )

    candidates["opposite_peak_nearby"] = flags
    candidates["opposite_peak_distance_px"] = distances

    candidates["review_status"] = np.where(
        flags,
        "opposite_sign_pair",
        "needs_review"
    )

    output_path = (
        output_dir / "D5_candidates_validated.csv"
    )

    candidates.write(
        output_path,
        format="csv",
        overwrite=True
    )

    print("\nCandidate Validation Results")
    print("----------------------------")

    print("Total candidates:", len(candidates))

    print(
        "Candidates with nearby opposite peak:",
        int(np.count_nonzero(flags))
    )

    print(
        "Candidates without nearby opposite peak:",
        int(np.count_nonzero(~flags))
    )

    print("\nSaved:", output_path)
    print("Candidate validation completed.")
