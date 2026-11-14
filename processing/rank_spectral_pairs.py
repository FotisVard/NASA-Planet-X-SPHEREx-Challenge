
import argparse
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy.interpolate import RegularGridInterpolator

from fits_loader import load_spherex_fits


BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw"
SEARCH_DIR = BASE_DIR / "data" / "searches"

INPUT_FILE = SEARCH_DIR / "validated_pairs.csv"
OUTPUT_FILE = SEARCH_DIR / "spectral_ranked_pairs.csv"


def filename_from_url(url):
    return Path(urlparse(str(url)).path).name


def wavelength_at_target(data, target):
    """
    Interpolate the first WCS-WAVE component
    directly at the target pixel position.

    Assumes X/Y are 1-based FITS coordinates.
    The physical meaning of component 0
    must be confirmed against SPHEREx documentation.
    """

    x = np.asarray(data["wave_x"], dtype=float) - 1
    y = np.asarray(data["wave_y"], dtype=float) - 1

    values = np.asarray(
        data["wavelength_grid"],
        dtype=float
    )

    wavelength_grid = values[:, :, 0]

    interpolator = RegularGridInterpolator(
        (y, x),
        wavelength_grid,
        bounds_error=False,
        fill_value=np.nan
    )

    px, py = data["wcs"].world_to_pixel(target)

    height, width = data["image"].shape

    if not (
        0 <= px < width
        and 0 <= py < height
    ):
        return np.nan

    return float(interpolator((float(py), float(px))))


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ra",
        type=float,
        default=133.795
    )

    parser.add_argument(
        "--dec",
        type=float,
        default=-7.245
    )

    parser.add_argument(
        "--tolerance-nm",
        type=float,
        default=5.0
    )

    args = parser.parse_args()

    target = SkyCoord(
        ra=args.ra * u.deg,
        dec=args.dec * u.deg,
        frame="icrs"
    )

    pairs = Table.read(INPUT_FILE, format="csv")

    cache = {}
    ranked = []
    skipped = 0

    for pair in pairs:

        url1 = str(pair["url_1"])
        url2 = str(pair["url_2"])

        path1 = RAW_DIR / filename_from_url(url1)
        path2 = RAW_DIR / filename_from_url(url2)

        if not path1.exists() or not path2.exists():
            skipped += 1
            continue

        for path in (path1, path2):

            if path not in cache:
                data = load_spherex_fits(path)
                cache[path] = wavelength_at_target(
                    data,
                    target
                )

        wave1 = cache[path1]
        wave2 = cache[path2]

        if not (
            np.isfinite(wave1)
            and np.isfinite(wave2)
        ):
            continue

        difference_nm = abs(wave2 - wave1) * 1000

        ranked.append({
            "detector": str(pair["detector"]),
            "obs_id_1": str(pair["obs_id_1"]),
            "obs_id_2": str(pair["obs_id_2"]),
            "delta_days": float(pair["delta_days"]),
            "wavelength_1_um": wave1,
            "wavelength_2_um": wave2,
            "difference_nm": difference_nm,
            "within_tolerance": (
                difference_nm <= args.tolerance_nm
            ),
            "url_1": url1,
            "url_2": url2
        })

    print("Total candidate pairs:", len(pairs))
    print("Pairs missing local FITS:", skipped)
    print("Pairs evaluated:", len(ranked))

    if not ranked:
        print("No pairs available for spectral evaluation.")
        return

    ranked.sort(
        key=lambda row: (
            row["difference_nm"],
            -row["delta_days"]
        )
    )

    result = Table(rows=ranked)

    result.write(
        OUTPUT_FILE,
        format="csv",
        overwrite=True
    )

    print("\nBest spectral matches:")

    result[
        [
            "detector",
            "delta_days",
            "wavelength_1_um",
            "wavelength_2_um",
            "difference_nm",
            "within_tolerance"
        ]
    ][:10].pprint(max_width=140)

    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
