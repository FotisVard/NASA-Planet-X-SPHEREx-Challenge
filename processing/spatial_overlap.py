
from pathlib import Path

import numpy as np
from astropy.table import Table
from spherical_geometry.polygon import SphericalPolygon


BASE_DIR = Path(__file__).resolve().parent.parent

SEARCH_DIR = BASE_DIR / "data" / "searches"

OBSERVATIONS_FILE = (
    SEARCH_DIR
    / "spherex_ra133.7950_dec-7.2450_r1.00arcmin.csv"
)

PAIRS_FILE = SEARCH_DIR / "candidate_pairs_auto.csv"

OUTPUT_FILE = (
    SEARCH_DIR / "candidate_pairs_spatial.csv"
)

TARGET_RA = 133.795
TARGET_DEC = -7.245


def parse_polygon(region):
    """
    Parse a simple ICRS POLYGON from SIA metadata.
    """

    parts = str(region).strip().split()

    if len(parts) < 10:
        return None

    if parts[0].upper() != "POLYGON":
        return None

    if parts[1].upper() != "ICRS":
        return None

    try:
        values = np.array(
            [float(x) for x in parts[2:]],
            dtype=float
        )

        if len(values) % 2 != 0:
            return None

        ra = values[0::2]
        dec = values[1::2]

        if len(ra) < 3:
            return None

        return SphericalPolygon.from_radec(
            ra,
            dec,
            degrees=True
        )

    except (ValueError, TypeError):
        return None


def contains_target(polygon, ra, dec):

    if polygon is None:
        return False

    return bool(
        polygon.contains_radec(
            ra,
            dec,
            degrees=True
        )
    )


def main():

    observations = Table.read(
        OBSERVATIONS_FILE,
        format="csv"
    )

    pairs = Table.read(
        PAIRS_FILE,
        format="csv"
    )

    print("Observations:", len(observations))
    print("Candidate pairs:", len(pairs))

    # URLs identify individual data products,
    # unlike obs_id, which may be repeated.
    coverage = {}

    for row in observations:

        url = str(row["access_url"])

        polygon = parse_polygon(
            row["s_region"]
        )

        coverage[url] = contains_target(
            polygon,
            TARGET_RA,
            TARGET_DEC
        )

    selected = []

    for row in pairs:

        url1 = str(row["url_1"])
        url2 = str(row["url_2"])

        if not coverage.get(url1, False):
            continue

        if not coverage.get(url2, False):
            continue

        selected.append(row)

    print(
        "Pairs covering target center:",
        len(selected)
    )

    if not selected:
        print("No matching pairs found.")
        return

    result = Table(rows=selected)

    result.write(
        OUTPUT_FILE,
        format="csv",
        overwrite=True
    )

    print("\nFirst 10 selected pairs:")

    result[
        [
            "obs_id_1",
            "obs_id_2",
            "delta_days",
            "em_min_um",
            "em_max_um"
        ]
    ][:10].pprint(max_width=120)

    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
