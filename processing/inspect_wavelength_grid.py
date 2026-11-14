
from pathlib import Path
import numpy as np
from astropy.io import fits

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

FILES = [
    "level2_2025W19_1B_0241_1D1_spx_l2b-v27-2026-197.fits",
    "level2_2026W19_1A_0594_2D1_spx_l2b-v27-2026-245.fits"
]


def inspect(filename):
    path = RAW_DIR / filename

    print("\n" + "=" * 65)
    print("FILE:", filename)

    with fits.open(path, memmap=True) as hdul:
        hdu = hdul["WCS-WAVE"]

        print("\nHDU type:", type(hdu).__name__)
        print("Columns:", hdu.columns.names)

        for col in hdu.columns:
            print(
                "Column:",
                col.name,
                "| format:",
                col.format,
                "| unit:",
                col.unit,
                "| dim:",
                col.dim
            )

        print("\nNumber of rows:", len(hdu.data))

        for name in hdu.columns.names:
            value = np.asarray(hdu.data[name][0])

            print(f"\n{name}")
            print("Shape:", value.shape)
            print("Dtype:", value.dtype)

            if np.issubdtype(value.dtype, np.number):
                finite = value[np.isfinite(value)]

                if finite.size:
                    print("Min:", float(np.min(finite)))
                    print("Max:", float(np.max(finite)))
                    print(
                        "First values:",
                        value.ravel()[:10]
                    )

        print("\nRelevant header keywords:")

        for key in hdu.header:
            if (
                key.startswith("TTYPE")
                or key.startswith("TFORM")
                or key.startswith("TDIM")
                or key.startswith("TUNIT")
            ):
                print(key, "=", hdu.header[key])


for filename in FILES:
    inspect(filename)
