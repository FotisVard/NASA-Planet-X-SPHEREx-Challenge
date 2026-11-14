
from pathlib import Path
from itertools import combinations

import numpy as np
from astropy.table import Table


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data/searches"
    / "spherex_ra133.7950_dec-7.2450_r1.00arcmin.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data/searches"
    / "candidate_pairs_auto.csv"
)


def spectral_group(row):
    """Return rounded spectral range in micrometers."""
    return (
        round(float(row["em_min"]) * 1e6, 3),
        round(float(row["em_max"]) * 1e6, 3)
    )


def select_pairs(table, min_days=180):
    pairs = []

    for i, j in combinations(range(len(table)), 2):
        first = table[i]
        second = table[j]

        if spectral_group(first) != spectral_group(second):
            continue

        time1 = float(first["t_min"])
        time2 = float(second["t_min"])

        if not np.isfinite(time1) or not np.isfinite(time2):
            continue

        delta_days = abs(time2 - time1)

        if delta_days < min_days:
            continue

        # Chronological ordering
        if time1 > time2:
            first, second = second, first

        band = spectral_group(first)

        pairs.append({
            "obs_id_1": str(first["obs_id"]),
            "obs_id_2": str(second["obs_id"]),
            "url_1": str(first["access_url"]),
            "url_2": str(second["access_url"]),
            "time_1_mjd": float(first["t_min"]),
            "time_2_mjd": float(second["t_min"]),
            "delta_days": float(delta_days),
            "em_min_um": band[0],
            "em_max_um": band[1],
            "ra_1": float(first["s_ra"]),
            "dec_1": float(first["s_dec"]),
            "ra_2": float(second["s_ra"]),
            "dec_2": float(second["s_dec"])
        })

    pairs.sort(
        key=lambda pair: pair["delta_days"],
        reverse=True
    )

    return pairs


def main():
    table = Table.read(INPUT_FILE, format="csv")

    print("Loaded observations:", len(table))

    pairs = select_pairs(table)

    print("Potential temporal pairs:", len(pairs))

    if not pairs:
        print("No pairs satisfy the current criteria.")
        return

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    result = Table(rows=pairs)
    result.write(
        OUTPUT_FILE,
        format="csv",
        overwrite=True
    )

    print("\nFirst 10 candidate pairs:")

    result[
        ["obs_id_1", "obs_id_2", "delta_days",
         "em_min_um", "em_max_um"]
    ][:10].pprint(max_width=120)

    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
