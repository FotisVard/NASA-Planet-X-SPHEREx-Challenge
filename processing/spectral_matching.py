
from pathlib import Path

import numpy as np
from astropy.table import Table
from scipy.interpolate import RegularGridInterpolator
from reproject import reproject_interp

from fits_loader import load_spherex_fits


def create_wavelength_map(data):
    """
    Create a full-resolution wavelength map
    from the SPHEREx WCS-WAVE lookup table.

    Wavelengths are measured in micrometers.
    """

    x = np.asarray(data["wave_x"], dtype=float)
    y = np.asarray(data["wave_y"], dtype=float)

    values = np.asarray(
        data["wavelength_grid"], dtype=float
    )

    wavelength = values[:, :, 0]

    if wavelength.shape != (len(y), len(x)):
        raise ValueError(
            "Unexpected WCS-WAVE grid dimensions."
        )

    interpolator = RegularGridInterpolator(
        (y - 1, x - 1),
        wavelength,
        bounds_error=False,
        fill_value=np.nan
    )

    height, width = data["image"].shape

    yy, xx = np.indices((height, width))

    coordinates = np.column_stack(
        (yy.ravel(), xx.ravel())
    )

    wavelength_map = interpolator(
        coordinates
    ).reshape(height, width)

    return wavelength_map


def compare_spectral_maps(data1, data2):
    """
    Compare wavelength maps in native pixel coordinates.

    This is a diagnostic comparison only.
    """

    wave1 = create_wavelength_map(data1)
    wave2 = create_wavelength_map(data2)

    if wave1.shape != wave2.shape:
        raise ValueError(
            "Wavelength maps have different shapes."
        )

    valid = (
        np.isfinite(wave1)
        & np.isfinite(wave2)
    )

    if not np.any(valid):
        raise ValueError(
            "No valid wavelength pixels found."
        )

    difference = np.abs(
        wave1[valid] - wave2[valid]
    )

    return {
        "median_difference_um": float(
            np.median(difference)
        ),
        "max_difference_um": float(
            np.max(difference)
        ),
        "valid_pixels": int(
            np.count_nonzero(valid)
        ),
    }


def compare_aligned_spectral_maps(data1, data2):
    """
    Align the second wavelength map to the first
    image's celestial WCS and compare wavelengths
    at corresponding sky positions.
    """

    wave1 = create_wavelength_map(data1)
    wave2 = create_wavelength_map(data2)

    aligned_wave2, footprint = reproject_interp(
        (wave2, data2["wcs"]),
        data1["wcs"],
        shape_out=wave1.shape
    )

    valid = (
        np.isfinite(wave1)
        & np.isfinite(aligned_wave2)
        & (footprint > 0)
    )

    if not np.any(valid):
        raise ValueError(
            "No valid overlap between wavelength maps."
        )

    difference = np.abs(
        wave1[valid] - aligned_wave2[valid]
    )

    return {
        "median_difference_um": float(
            np.median(difference)
        ),
        "max_difference_um": float(
            np.max(difference)
        ),
        "valid_pixels": int(
            np.count_nonzero(valid)
        ),
        "overlap_percentage": float(
            100 * np.count_nonzero(valid) / wave1.size
        ),
    }


if __name__ == "__main__":

    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / "data" / "raw"

    pairs_path = (
        base_dir / "data" / "candidate_pairs.csv"
    )

    if not pairs_path.is_file():
        raise FileNotFoundError(
            f"Candidate pairs file not found: {pairs_path}"
        )

    pairs = Table.read(
        pairs_path,
        format="csv"
    )

    if len(pairs) == 0:
        raise ValueError(
            "No candidate pairs found."
        )

    pair = pairs[0]

    filename1 = str(
        pair["url_1"]
    ).split("/")[-1]

    filename2 = str(
        pair["url_2"]
    ).split("/")[-1]

    print("\nLoading SPHEREx observations...")
    print("Image 1:", filename1)
    print("Image 2:", filename2)

    data1 = load_spherex_fits(
        raw_dir / filename1
    )

    data2 = load_spherex_fits(
        raw_dir / filename2
    )

    # -----------------------------------
    # 1. Original spectral comparison
    # -----------------------------------

    original_result = compare_spectral_maps(
        data1, data2
    )

    print("\nOriginal Spectral Matching")
    print("--------------------------")

    print(
        "Median wavelength difference:",
        original_result["median_difference_um"],
        "um"
    )

    print(
        "Maximum wavelength difference:",
        original_result["max_difference_um"],
        "um"
    )

    print(
        "Valid pixels:",
        original_result["valid_pixels"]
    )

    # -----------------------------------
    # 2. Aligned spectral comparison
    # -----------------------------------

    aligned_result = compare_aligned_spectral_maps(
        data1, data2
    )

    print("\nAligned Spectral Matching")
    print("-------------------------")

    print(
        "Median wavelength difference:",
        aligned_result["median_difference_um"],
        "um"
    )

    print(
        "Maximum wavelength difference:",
        aligned_result["max_difference_um"],
        "um"
    )

    print(
        "Valid overlapping pixels:",
        aligned_result["valid_pixels"]
    )

    print(
        "Overlap percentage:",
        round(
            aligned_result["overlap_percentage"], 2
        ),
        "%"
    )

    print("\nSpectral analysis completed.")
