
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u
from scipy.ndimage import map_coordinates

from fits_loader import load_spherex_fits
from spectral_matching import create_wavelength_map


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


def main():

    wavelengths = []

    for filename in FILES:

        data = load_spherex_fits(RAW_DIR / filename)

        wave_map = create_wavelength_map(data)

        x, y = data["wcs"].world_to_pixel(TARGET)

        wavelength = float(
            map_coordinates(
                wave_map,
                [[float(y)], [float(x)]],
                order=1,
                mode="constant",
                cval=np.nan
            )[0]
        )

        print("\nFile:", filename)
        print("Pixel X:", round(float(x), 3))
        print("Pixel Y:", round(float(y), 3))
        print("Wavelength:", wavelength, "um")

        wavelengths.append(wavelength)

    if np.all(np.isfinite(wavelengths)):
        difference = abs(
            wavelengths[1] - wavelengths[0]
        )

        print("\nTarget wavelength difference:")
        print(round(difference, 6), "um")
        print(round(difference * 1000, 3), "nm")


if __name__ == "__main__":
    main()
