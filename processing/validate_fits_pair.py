
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"

FILES = [
    "level2_2025W19_1B_0241_1D1_spx_l2b-v27-2026-197.fits",
    "level2_2026W19_1A_0594_2D1_spx_l2b-v27-2026-245.fits"
]

TARGET = SkyCoord(
    ra=133.795 * u.deg,
    dec=-7.245 * u.deg,
    frame="icrs"
)


def inspect_file(filename):

    path = RAW_DIR / filename

    print("\n" + "=" * 65)
    print("File:", filename)
    print("Size:", round(path.stat().st_size / 1024**2, 2), "MB")

    with fits.open(path, memmap=True) as hdul:

        hdul.verify("exception")

        print("\nFITS extensions:")

        for hdu in hdul:
            shape = None if hdu.data is None else hdu.data.shape
            print(hdu.name, shape)

        image_hdu = next(
            (
                hdu for hdu in hdul
                if hdu.name.upper() == "IMAGE"
                and hdu.data is not None
            ),
            None
        )

        if image_hdu is None:
            raise ValueError("IMAGE extension not found.")

        image = image_hdu.data

        print("\nImage shape:", image.shape)

        finite = np.isfinite(image)

        print("Finite pixels:", np.count_nonzero(finite))
        print("Finite percentage:", round(100 * finite.mean(), 2))

        if not finite.any():
            raise ValueError("No finite image pixels.")

        print("Median brightness:", float(np.median(image[finite])))

        wcs = WCS(image_hdu.header).celestial

        x, y = wcs.world_to_pixel(TARGET)

        inside = (
            0 <= x < image.shape[1]
            and 0 <= y < image.shape[0]
        )

        print("\nTarget pixel X:", round(float(x), 2))
        print("Target pixel Y:", round(float(y), 2))
        print("Target inside image:", inside)

        if inside:
            xi = int(round(float(x)))
            yi = int(round(float(y)))

            if 0 <= xi < image.shape[1] and 0 <= yi < image.shape[0]:
                print(
                    "Target pixel finite:",
                    bool(np.isfinite(image[yi, xi]))
                )


def main():

    for filename in FILES:
        inspect_file(filename)

    print("\nFITS inspection completed.")


if __name__ == "__main__":
    main()
