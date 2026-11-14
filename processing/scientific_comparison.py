"""Web-safe wrapper around the project's existing scientific analysis modules."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

from candidate_detection import detect_candidates, save_candidate_figure
from candidate_quality import assess_candidate, get_cutout
from fits_loader import load_spherex_fits
from image_comparison import compare_images
from motion_candidates import find_motion_pairs, plot_motion_pairs


def _limits(values, percentiles=(2, 98), symmetric=False):
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return (-1.0, 1.0)
    if symmetric:
        limit = float(np.percentile(np.abs(finite), percentiles[1]))
        return (-max(limit, 1e-12), max(limit, 1e-12))
    low, high = np.percentile(finite, percentiles)
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def run_scientific_comparison(
    first_path, second_path, output_dir,
    wavelength_tolerance=0.005, sigma_threshold=5.0,
):
    """Run existing FLAGS/VARIANCE/WCS-WAVE analysis and save diagnostics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nSPHEREx Scientific Comparison")
    print("=============================")
    print("Loading with processing/fits_loader.py")
    first = load_spherex_fits(first_path)
    second = load_spherex_fits(second_path)
    print("First:", first["metadata"]["date_obs"], first["metadata"]["filename"])
    print("Second:", second["metadata"]["date_obs"], second["metadata"]["filename"])
    print("ZODI subtraction: enabled")
    print("Quality flags: enabled via processing/quality_control.py")
    print("Variance significance: enabled")
    print("Wavelength tolerance:", wavelength_tolerance, "um")
    print("Candidate threshold:", sigma_threshold, "sigma")
    print("Running processing/image_comparison.py")

    result = compare_images(first, second, wavelength_tolerance=wavelength_tolerance)
    valid = result["valid_mask"]
    valid_count = int(np.count_nonzero(valid))
    total_count = int(valid.size)
    if valid_count == 0:
        raise RuntimeError("No spectrally compatible quality pixels remain for comparison.")

    difference = result["difference"]
    background_offset = float(np.nanmedian(difference[valid]))
    corrected_difference = np.full_like(difference, np.nan)
    corrected_difference[valid] = difference[valid] - background_offset
    significance = result["significance"]
    spectral = result["spectral_difference"][valid]

    summary = {
        "valid_pixels": valid_count,
        "total_pixels": total_count,
        "valid_percentage": 100.0 * valid_count / total_count,
        "median_wavelength_difference_um": float(np.nanmedian(spectral)),
        "max_wavelength_difference_um": float(np.nanmax(spectral)),
        "background_offset_mjy_sr": background_offset,
        "significant_pixels_abs_gt_5": int(np.count_nonzero(np.abs(significance[valid]) > 5)),
        "wavelength_tolerance_um": wavelength_tolerance,
        "sigma_threshold": sigma_threshold,
    }

    print("\nScientific summary")
    print("------------------")
    for key, value in summary.items():
        print(f"{key}: {value}")

    image_limits = _limits(np.concatenate([result["image1"][valid], result["aligned_image2"][valid]]))
    difference_limits = _limits(corrected_difference[valid], symmetric=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    panels = [
        (result["image1"], "First observation (ZODI subtracted)", "gray", image_limits),
        (result["aligned_image2"], "Second observation (WCS aligned)", "gray", image_limits),
        (corrected_difference, "Background-corrected difference", "coolwarm", difference_limits),
        (significance, "Difference significance (σ)", "coolwarm", (-5, 5)),
    ]
    for axis, (image, title, cmap, limits) in zip(axes.flat, panels):
        shown = axis.imshow(image, origin="lower", cmap=cmap, vmin=limits[0], vmax=limits[1])
        axis.set_title(title)
        axis.axis("off")
        fig.colorbar(shown, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    figure_path = output_dir / "scientific_comparison.png"
    fig.savefig(figure_path, dpi=140, bbox_inches="tight")
    plt.close(fig)

    summary_path = output_dir / "scientific_summary.json"
    print("\nRunning processing/candidate_detection.py")
    candidates, candidate_score = detect_candidates(
        corrected_difference, valid, threshold=float(sigma_threshold),
        border_distance=max(1, min(8, min(valid.shape) // 10)),
    )
    candidates = candidates[:50]
    print("Candidate peaks:", len(candidates))
    candidate_overview = output_dir / "candidate_overview.png"
    save_candidate_figure(corrected_difference, candidates, candidate_overview, max_display=50)

    print("Running processing/motion_candidates.py")
    motion_pairs = (
        find_motion_pairs(Table(rows=candidates), max_distance=10.0, min_distance=0.75)
        if candidates else []
    )
    print("Motion-pair hypotheses:", len(motion_pairs))
    motion_figure = None
    if motion_pairs:
        motion_path = output_dir / "motion_hypotheses.png"
        plot_motion_pairs(corrected_difference, motion_pairs, motion_path, max_pairs=20)
        motion_figure = motion_path.name

    motion_keys = set()
    for pair in motion_pairs:
        motion_keys.add((pair["negative_x"], pair["negative_y"]))
        motion_keys.add((pair["positive_x"], pair["positive_y"]))

    print("Running processing/candidate_quality.py")
    candidate_cards = []
    for index, candidate in enumerate(candidates[:12], 1):
        x, y = int(candidate["x"]), int(candidate["y"])
        quality = assess_candidate(
            result["image1"], result["aligned_image2"],
            corrected_difference, valid, candidate, radius=8,
        )
        ra, dec = first["wcs"].pixel_to_world_values(x, y)
        if (x, y) in motion_keys:
            classification = "possible_moving_source"
        elif candidate["polarity"] == "positive":
            classification = "brightened_or_transient"
        else:
            classification = "faded_or_disappeared"

        card_path = output_dir / f"candidate_{index:02d}.png"
        cutouts = [
            get_cutout(result["image1"], x, y, 8),
            get_cutout(result["aligned_image2"], x, y, 8),
            get_cutout(corrected_difference, x, y, 8),
            get_cutout(significance, x, y, 8),
        ]
        titles = ["Epoch A", "Epoch B", "Difference", "Significance"]
        fig, axes = plt.subplots(1, 4, figsize=(9, 2.5))
        for axis, cutout, title in zip(axes, cutouts, titles):
            axis.imshow(cutout, origin="lower", cmap="gray" if title.startswith("Epoch") else "coolwarm")
            axis.set_title(title, fontsize=9)
            axis.axis("off")
        fig.suptitle(f"Candidate {index} · {candidate['score']:.1f} local σ", fontsize=10)
        fig.tight_layout()
        fig.savefig(card_path, dpi=130, bbox_inches="tight")
        plt.close(fig)

        candidate_cards.append({
            "rank": index, "x": x, "y": y,
            "ra": float(ra), "dec": float(dec),
            "polarity": candidate["polarity"],
            "score": float(candidate["score"]),
            "propagated_significance": float(significance[y, x]),
            "classification": classification,
            "quality_flag": quality["quality_flag"],
            "component_pixels": quality["component_pixels"],
            "valid_fraction": quality["valid_fraction"],
            "figure": card_path.name,
        })

    summary["candidate_count"] = len(candidates)
    summary["motion_pair_count"] = len(motion_pairs)
    summary["candidate_overview"] = candidate_overview.name
    summary["motion_figure"] = motion_figure
    summary["candidates"] = candidate_cards
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Saved:", candidate_overview)
    if motion_figure:
        print("Saved:", output_dir / motion_figure)
    print("Saved:", figure_path)
    print("Saved:", summary_path)
    return {**summary, "figure": figure_path.name, "summary_file": summary_path.name}
