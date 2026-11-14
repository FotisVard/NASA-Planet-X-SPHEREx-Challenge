
from pathlib import Path

import numpy as np

from astropy.io import fits
from astropy.wcs import WCS


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

FILES = [
    "level2_2025W19_1B_0181_3D5_spx_l2b-v27-2026-197.fits",
    "level2_2026W20_1A_1166_1D5_spx_l2b-v27-2026-246.fits",
]


def inspect_file(path):
    print("\n" + "=" * 65)
    print("FILE:", path.name)
    print("=" * 65)

    if not path.exists():
        print("ERROR: File not found.")
        return

    with fits.open(path, memmap=True) as hdul:

        print("\nFITS HDUs:")
        hdul.info()

        for index, hdu in enumerate(hdul):
            header = hdu.header

            print("\n" + "-" * 50)
            print(f"HDU {index}: {hdu.name}")
            print("-" * 50)

            if hdu.data is not None:
                print("Data shape:", hdu.data.shape)
            else:
                print("No image data")

            keywords = [
                "CTYPE1", "CTYPE2",
                "CRVAL1", "CRVAL2",
                "CRPIX1", "CRPIX2",
                "CDELT1", "CDELT2",
                "CD1_1", "CD1_2",
                "CD2_1", "CD2_2",
                "PC1_1", "PC1_2",
                "PC2_1", "PC2_2",
                "RADESYS", "EQUINOX",
                "DATE-OBS", "MJD-OBS",
            ]

            print("\nRelevant FITS keywords:")

            found = False

            for keyword in keywords:
                if keyword in header:
                    print(
                        f"{keyword:12s} = {header[keyword]}"
                    )
                    found = True

            if not found:
                print("No standard WCS keywords found.")

            if hdu.data is None:
                continue

            if not isinstance(hdu.data, np.ndarray):
                continue

            if hdu.data.ndim != 2:
                print(
                    "Skipping coordinate test: "
                    "data is not a 2D image."
                )
                continue

            try:
                wcs = WCS(header, naxis=2)

                if not wcs.has_celestial:
                    print(
                        "\nNo celestial WCS detected "
                        "in this HDU."
                    )
                    continue

                height, width = hdu.data.shape

                center_x = (width - 1) / 2
                center_y = (height - 1) / 2

                ra, dec = wcs.pixel_to_world_values(
                    center_x, center_y
                )

                print("\nCelestial WCS detected!")
                print(
                    f"Image center: "
                    f"x={center_x:.2f}, "
                    f"y={center_y:.2f}"
                )
                print(f"RA:  {ra:.8f} deg")
                print(f"Dec: {dec:.8f} deg")

                test_x, test_y = (
                    wcs.world_to_pixel_values(ra, dec)
                )

                error = np.hypot(
                    test_x - center_x,
                    test_y - center_y,
                )

                print(
                    f"Round-trip error: {error:.8e} px"
                )

            except Exception as exc:
                print(
                    "\nWCS parsing failed:",
                    type(exc).__name__,
                    str(exc),
                )


def main():
    print("\nSPHEREx WCS Inspection")
    print("=====================")

    for filename in FILES:
        inspect_file(RAW_DIR / filename)

    print("\nInspection completed.")


if __name__ == "__main__":
    main()
