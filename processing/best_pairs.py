
from pathlib import Path
from astropy.table import Table


BASE_DIR = Path(__file__).resolve().parent.parent

SEARCH_DIR = BASE_DIR / "data" / "searches"

INPUT_FILE = SEARCH_DIR / "validated_pairs.csv"

OUTPUT_FILE = SEARCH_DIR / "best_pairs.csv"


def select_best_pairs(table, max_per_detector=1):

    selected = []

    detectors = sorted(
        set(str(x) for x in table["detector"])
    )

    for detector in detectors:

        candidates = [
            row for row in table
            if str(row["detector"]) == detector
        ]

        candidates.sort(
            key=lambda row: float(row["delta_days"]),
            reverse=True
        )

        selected.extend(
            candidates[:max_per_detector]
        )

    return selected


def main():

    table = Table.read(
        INPUT_FILE,
        format="csv"
    )

    print("Validated pairs:", len(table))

    selected = select_best_pairs(
        table,
        max_per_detector=1
    )

    print("Selected pairs:", len(selected))

    if not selected:
        print("No pairs selected.")
        return

    result = Table(rows=selected)

    result.write(
        OUTPUT_FILE,
        format="csv",
        overwrite=True
    )

    print("\nSelected pairs:\n")

    result[
        [
            "detector",
            "obs_id_1",
            "obs_id_2",
            "delta_days"
        ]
    ].pprint(max_width=140)

    print("\nSaved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
