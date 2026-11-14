
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from photutils.detection import DAOStarFinder
from astropy.stats import sigma_clipped_stats
from scipy.spatial import cKDTree

from fits_loader import load_spherex_fits
from image_comparison import compare_images
from astrometric_analysis import fit_source


def robust_sigma(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 5:
        return np.nan

    median = np.median(values)

    return float(
        1.4826 * np.median(np.abs(values - median))
    )


def detect_reference_sources(
    image,
    valid_mask,
    max_sources=250,
):
    """Detect candidate reference stars."""

    finite_valid = valid_mask & np.isfinite(image)

    _, median, std = sigma_clipped_stats(
        image[finite_valid],
        sigma=3.0,
        maxiters=5,
    )

    detection_image = np.where(
        finite_valid,
        image - median,
        0.0,
    )

    finder = DAOStarFinder(
        fwhm=2.0,
        threshold=8.0 * std,
    )

    sources = finder(detection_image)

    if sources is None:
        return []

    print("DAOStarFinder columns:", sources.colnames)
    print("Detected sources:", len(sources))

    sources.sort("flux")
    sources.reverse()

    selected = []

    for source in sources:

        x = float(source["x_centroid"])
        y = float(source["y_centroid"])

        # Avoid the outermost image boundary.
        if (
            x < 20
            or y < 20
            or x >= image.shape[1] - 20
            or y >= image.shape[0] - 20
        ):
            continue

        ix = int(round(x))
        iy = int(round(y))

        local_mask = valid_mask[
            iy - 8:iy + 9,
            ix - 8:ix + 9,
        ]

        if local_mask.shape != (17, 17):
            continue

        if np.mean(local_mask) < 0.98:
            continue

        # Keep selected sources separated.
        if any(
            np.hypot(x - sx, y - sy) < 15
            for sx, sy in selected
        ):
            continue

        selected.append((x, y))

        if len(selected) >= max_sources:
            break

    return selected



def measure_reference_sources(
    image1,
    image2,
    valid_mask,
    reference_positions,
):
    rows = []

    rejected_fit = 0
    rejected_bounds = 0
    rejected_width = 0

    # Store Gaussian widths for diagnostics
    all_widths = []

    for index, (x, y) in enumerate(
        reference_positions, 1
    ):

        fit1 = fit_source(
            image1,
            valid_mask,
            x,
            y,
            radius=6,
        )

        fit2 = fit_source(
            image2,
            valid_mask,
            x,
            y,
            radius=6,
        )

        if fit1 is None or fit2 is None:
            rejected_fit += 1
            continue

        if (
            fit1["at_position_bound"]
            or fit2["at_position_bound"]
        ):
            rejected_bounds += 1
            continue

        widths = [
            fit1["sigma_x"],
            fit1["sigma_y"],
            fit2["sigma_x"],
            fit2["sigma_y"],
        ]

        # Record widths before applying the filter
        all_widths.append(widths)

        if not all(
            0.5 <= width < 3.0
            for width in widths
        ):
            rejected_width += 1
            continue

        dx = fit2["x"] - fit1["x"]
        dy = fit2["y"] - fit1["y"]

        rows.append({
            "reference_id": index,
            "x": x,
            "y": y,
            "dx": float(dx),
            "dy": float(dy),
            "displacement": float(
                np.hypot(dx, dy)
            ),
            "fit_rms_2025": fit1["fit_rms"],
            "fit_rms_2026": fit2["fit_rms"],
        })

    # Reference source quality report
    print("\nReference Source Quality Report")
    print("-------------------------------")
    print("Input positions:", len(reference_positions))
    print("Rejected: fit failed:", rejected_fit)
    print("Rejected: position bounds:", rejected_bounds)
    print("Rejected: Gaussian width:", rejected_width)
    print("Accepted:", len(rows))

    # Gaussian width diagnostics
    if all_widths:
        width_array = np.asarray(
            all_widths,
            dtype=float,
        )

        print("\nGaussian Width Diagnostics")
        print("--------------------------")

        labels = [
            "sigma_x_2025",
            "sigma_y_2025",
            "sigma_x_2026",
            "sigma_y_2026",
        ]

        for i, label in enumerate(labels):
            values = width_array[:, i]

            print(
                f"{label}: "
                f"min={np.min(values):.4f}, "
                f"median={np.median(values):.4f}, "
                f"max={np.max(values):.4f}"
            )

        print(
            "Sources with sigma <= 0.6:",
            np.sum(
                np.any(
                    width_array <= 0.6,
                    axis=1,
                )
            )
        )

        print(
            "Sources with sigma >= 3.0:",
            np.sum(
                np.any(
                    width_array >= 3.0,
                    axis=1,
                )
            )
        )

    return rows



def evaluate_candidates(
    candidates,
    references,
    neighbor_count=25,
    max_radius=300.0,
):
    """
    Compare each candidate's displacement vector
    with nearby reference-source displacement vectors.

    The resulting residual is not a formal
    statistical significance.
    """

    reference_positions = np.column_stack([
        references["x"],
        references["y"],
    ])

    tree = cKDTree(reference_positions)

    results = []

    for candidate in candidates:

        cx = 0.5 * (
            float(candidate["negative_x"])
            + float(candidate["positive_x"])
        )

        cy = 0.5 * (
            float(candidate["negative_y"])
            + float(candidate["positive_y"])
        )

        distances, indices = tree.query(
            [cx, cy],
            k=min(neighbor_count, len(references)),
            distance_upper_bound=max_radius,
        )

        distances = np.atleast_1d(distances)
        indices = np.atleast_1d(indices)

        valid_neighbors = indices < len(references)
        indices = indices[valid_neighbors]

        if len(indices) < 8:
            results.append({
                "pair_id": int(candidate["pair_id"]),
                "reference_count": len(indices),
                "local_dx": np.nan,
                "local_dy": np.nan,
                "corrected_dx": np.nan,
                "corrected_dy": np.nan,
                "corrected_displacement": np.nan,
                "local_sigma": np.nan,
                "relative_score": np.nan,
            })
            continue

        local_dx = np.asarray(
            references["dx"][indices],
            dtype=float,
        )

        local_dy = np.asarray(
            references["dy"][indices],
            dtype=float,
        )

        median_dx = float(np.median(local_dx))
        median_dy = float(np.median(local_dy))

        residual_dx = local_dx - median_dx
        residual_dy = local_dy - median_dy

        sigma_x = robust_sigma(residual_dx)
        sigma_y = robust_sigma(residual_dy)

        # Descriptive 2D scatter scale, not a formal
        # uncertainty of the candidate centroid.
        local_sigma = float(
            np.hypot(sigma_x, sigma_y)
        )

        candidate_dx = float(
            candidate["dx_pixels"]
        )

        candidate_dy = float(
            candidate["dy_pixels"]
        )

        corrected_dx = candidate_dx - median_dx
        corrected_dy = candidate_dy - median_dy

        corrected_displacement = float(
            np.hypot(corrected_dx, corrected_dy)
        )

        if (
            np.isfinite(local_sigma)
            and local_sigma > 0
        ):
            relative_score = (
                corrected_displacement / local_sigma
            )
        else:
            relative_score = np.nan

        results.append({
            "pair_id": int(candidate["pair_id"]),
            "reference_count": len(indices),
            "local_dx": median_dx,
            "local_dy": median_dy,
            "corrected_dx": corrected_dx,
            "corrected_dy": corrected_dy,
            "corrected_displacement":
                corrected_displacement,
            "local_sigma": local_sigma,
            "relative_score": relative_score,
        })

    return results


if __name__ == "__main__":

    base_dir = Path(__file__).resolve().parent.parent

    raw_dir = base_dir / "data" / "raw"

    processed_dir = (
        base_dir / "data" / "processed" / "automatic"
    )

    print("\nSPHEREx Astrometric Validation")
    print("=============================")

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

    print("\nDetecting reference sources...")

    positions = detect_reference_sources(
        image1,
        valid_mask,
        max_sources=1000,
    )

    print("Detected reference positions:", len(positions))

    print("\nMeasuring reference centroids...")

    reference_rows = measure_reference_sources(
        image1,
        image2,
        valid_mask,
        positions,
    )

    print("Accepted reference sources:", len(reference_rows))

    if len(reference_rows) < 8:
        raise RuntimeError(
            "Not enough valid reference sources."
        )

    references = Table(rows=reference_rows)

    reference_path = (
        processed_dir / "D5_reference_astrometry.csv"
    )

    references.write(
        reference_path,
        format="csv",
        overwrite=True,
    )

    global_dx = float(np.median(references["dx"]))
    global_dy = float(np.median(references["dy"]))

    scatter_dx = robust_sigma(references["dx"])
    scatter_dy = robust_sigma(references["dy"])

    print("\nReference Astrometry")
    print("--------------------")
    print("Median dx:", global_dx)
    print("Median dy:", global_dy)
    print("Robust scatter dx:", scatter_dx)
    print("Robust scatter dy:", scatter_dy)

    candidates = Table.read(
        processed_dir / "D5_astrometric_results.csv",
        format="csv",
    )

    results = evaluate_candidates(
        candidates,
        references,
        neighbor_count=25,
        max_radius=300.0,
    )

    validation = Table(rows=results)

    validation_path = (
        processed_dir / "D5_astrometric_validation.csv"
    )

    validation.write(
        validation_path,
        format="csv",
        overwrite=True,
    )

    print("\nCandidate Validation Results")
    print("----------------------------")

    for row in validation:
        print(
            f"Pair {row['pair_id']}: "
            f"references={row['reference_count']} "
            f"corrected_displacement="
            f"{row['corrected_displacement']:.4f}px "
            f"relative_score="
            f"{row['relative_score']:.2f}"
        )

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.scatter(
        references["dx"],
        references["dy"],
        s=15,
        alpha=0.5,
        label="Reference sources",
    )

    ax.scatter(
        candidates["dx_pixels"],
        candidates["dy_pixels"],
        s=75,
        marker="x",
        label="Motion candidates",
    )

    ax.axhline(0, color="gray", linewidth=0.7)
    ax.axvline(0, color="gray", linewidth=0.7)

    ax.set_xlabel("dx (pixels)")
    ax.set_ylabel("dy (pixels)")
    ax.set_title("SPHEREx-D5 Astrometric Displacements")
    ax.legend()
    ax.grid(alpha=0.2)

    plt.tight_layout()

    figure_path = (
        processed_dir / "D5_astrometric_validation.png"
    )

    plt.savefig(
        figure_path,
        dpi=150,
    )

    plt.close(fig)

    print("\nSaved references:", reference_path)
    print("Saved validation:", validation_path)
    print("Saved figure:", figure_path)

    print("\nAstrometric validation completed.")
