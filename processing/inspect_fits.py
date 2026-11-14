
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.table import Table

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

pairs = Table.read(
    BASE_DIR / "data" / "candidate_pairs.csv",
    format="csv"
)

pair = pairs[0]

for index, url_column in enumerate(["url_1", "url_2"], start=1):

    filename = str(pair[url_column]).split("/")[-1]
    filepath = RAW_DIR / filename

    print(f"\n{'=' * 60}")
    print(f"IMAGE {index}: {filename}")
    print(f"{'=' * 60}")

    with fits.open(filepath, memmap=True) as hdul:

        print("\nFITS EXTENSIONS:")
        hdul.info()

        image = hdul["IMAGE"].data
        variance = hdul["VARIANCE"].data
        flags = hdul["FLAGS"].data

        valid_image = np.isfinite(image)
        valid_variance = (
            np.isfinite(variance) & (variance > 0)
        )

        print("\nIMAGE STATISTICS:")
        print("Shape:", image.shape)
        print("Unit:", hdul["IMAGE"].header.get("BUNIT"))
        print("Finite pixels:", np.count_nonzero(valid_image))
        print("Median brightness:",
              np.nanmedian(image))

        print("\nVARIANCE STATISTICS:")
        print("Shape:", variance.shape)
        print("Positive finite variance pixels:",
              np.count_nonzero(valid_variance))
        print("Median positive variance:",
              np.median(variance[valid_variance])
              if np.any(valid_variance) else "N/A")

        print("\nFLAGS STATISTICS:")
        print("Shape:", flags.shape)

        values, counts = np.unique(
            flags, return_counts=True
        )

        top_indices = np.argsort(counts)[-10:][::-1]

        for idx in top_indices:
            print(
                f"Flag value {values[idx]}: "
                f"{counts[idx]} pixels"
            )

        print("\nWCS-WAVE:")
        if "WCS-WAVE" in hdul:
            wave = hdul["WCS-WAVE"]
            print("Columns:", wave.columns.names)
            print("Rows:", len(wave.data))
            print("Header:")
            print(repr(wave.header))
        else:
            print("WCS-WAVE extension not found.")
