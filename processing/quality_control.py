
from pathlib import Path

import numpy as np
from fits_loader import load_spherex_fits


# Initial quality policy for our prototype
BAD_FLAG_NAMES = {
    "TRANSIENT",
    "OVERFLOW",
    "SUR_ERROR",
    "NONFUNC",
    "MISSING_DATA",
    "HOT",
    "COLD",
    "PHANMISS",
    "NONLINEAR",
    "PERSIST",
    "OUTLIER",
    "CROSSTALK",
    "GHOST",
    "GHOST_FPA",
    "GHOST_EXT",
    "STREAK",
    "BLOOM",
    "SNOWBALL",
    "HALO",
}


def get_bad_flag_mask(header):
    """
    Build an integer bitmask from the FITS flag definitions.
    """

    bad_mask = 0
    excluded_names = []

    for key in header:
        if not key.startswith("MSKN"):
            continue

        suffix = key[4:]
        name = str(header[key]).strip()

        if name in BAD_FLAG_NAMES:
            mask_key = "MSKM" + suffix

            if mask_key in header:
                bad_mask |= int(header[mask_key])
                excluded_names.append(name)

    return bad_mask, excluded_names


def create_quality_mask(data):
    """
    Return a boolean array:
    True = pixel passes initial quality checks
    False = pixel excluded
    """

    image = data["image"]
    variance = data["variance"]
    flags = data["flags"]

    bad_mask, excluded_names = get_bad_flag_mask(
        data["flags_header"]
    )

    finite_mask = (
        np.isfinite(image)
        & np.isfinite(variance)
        & (variance > 0)
    )

    flag_mask = (flags & bad_mask) == 0

    quality_mask = finite_mask & flag_mask

    return quality_mask, excluded_names


if __name__ == "__main__":

    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / "data" / "raw"

    files = sorted(raw_dir.glob("*.fits"))

    if not files:
        raise FileNotFoundError(
            "No FITS files found in data/raw."
        )

    data = load_spherex_fits(files[0])

    quality_mask, excluded_flags = create_quality_mask(data)

    total_pixels = quality_mask.size
    good_pixels = np.count_nonzero(quality_mask)
    bad_pixels = total_pixels - good_pixels

    print("\nSPHEREx Quality Control")
    print("-----------------------")

    print("File:", data["metadata"]["filename"])
    print("Total pixels:", total_pixels)
    print("Good pixels:", good_pixels)
    print("Excluded pixels:", bad_pixels)

    print(
        "Good percentage:",
        round(100 * good_pixels / total_pixels, 2),
        "%"
    )

    print("\nExcluded flag types:")

    for name in excluded_flags:
        print("-", name)
