
from pathlib import Path
from collections import Counter

from astropy.table import Table


BASE_DIR = Path(__file__).resolve().parent.parent
SEARCH_DIR = BASE_DIR / "data" / "searches"

OBS_FILE = (
    SEARCH_DIR
    / "spherex_ra133.7950_dec-7.2450_r1.00arcmin.csv"
)

PAIRS_FILE = SEARCH_DIR / "candidate_pairs_spatial.csv"

OUTPUT_FILE = SEARCH_DIR / "validated_pairs.csv"


def main():

    observations = Table.read(
        OBS_FILE,
        format="csv"
    )

    pairs = Table.read(
        PAIRS_FILE,
        format="csv"
    )

    # Identify products by URL, not obs_id.
    metadata = {
        str(row["access_url"]): row
        for row in observations
    }

    validated = []
    rejected = Counter()

    for pair in pairs:

        url1 = str(pair["url_1"])
        url2 = str(pair["url_2"])

        if url1 == url2:
            rejected["same_file"] += 1
            continue

        if url1 not in metadata or url2 not in metadata:
            rejected["missing_metadata"] += 1
            continue

        first = metadata[url1]
        second = metadata[url2]

        band1 = str(first["energy_bandpassname"])
        band2 = str(second["energy_bandpassname"])

        if band1 != band2:
            rejected["different_detector"] += 1
            continue

        if int(first["calib_level"]) != int(second["calib_level"]):
            rejected["different_calibration"] += 1
            continue

        if (
            str(first["dataproduct_subtype"]).lower() != "science"
            or str(second["dataproduct_subtype"]).lower() != "science"
        ):
            rejected["not_science"] += 1
            continue

        if float(pair["delta_days"]) <= 0:
            rejected["same_time"] += 1
            continue

        validated.append({
            "obs_id_1": str(pair["obs_id_1"]),
            "obs_id_2": str(pair["obs_id_2"]),
            "detector": band1,
            "url_1": url1,
            "url_2": url2,
            "delta_days": float(pair["delta_days"]),
            "em_min_um": float(pair["em_min_um"]),
            "em_max_um": float(pair["em_max_um"])
        })

    print("Initial pairs:", len(pairs))
    print("Validated pairs:", len(validated))

    print("\nRejection reasons:")
    for reason, count in rejected.items():
        print(f"{reason}: {count}")

    if not validated:
        print("\nNo validated pairs.")
        return

    result = Table(rows=validated)

    result.write(
        OUTPUT_FILE,
        format="csv",
        overwrite=True
    )

    print("\nValidated pairs per detector:")
    for detector, count in sorted(
        Counter(result["detector"]).items()
    ):
        print(f"{detector}: {count}")

    print("\nFirst 10 validated pairs:")
    result[
        ["obs_id_1", "obs_id_2", "detector", "delta_days"]
    ][:10].pprint(max_width=120)

    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
