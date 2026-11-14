
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


def load_spherex_fits(filepath):
    """
    Load a SPHEREx Level-2 FITS file.

    Returns a dictionary containing the image,
    flags, variance, celestial WCS, wavelength
    calibration data, flag definitions and metadata.
    """

    filepath = Path(filepath)

    if not filepath.is_file():
        raise FileNotFoundError(
            f"FITS file not found: {filepath}"
        )

    with fits.open(filepath, memmap=False) as hdul:

        required = ["IMAGE", "FLAGS", "VARIANCE", "WCS-WAVE"]

        for name in required:
            if name not in hdul:
                raise ValueError(
                    f"Missing FITS extension: {name}"
                )

        image = np.asarray(
            hdul["IMAGE"].data, dtype=np.float32
        ).copy()

        flags = np.asarray(
            hdul["FLAGS"].data, dtype=np.int32
        ).copy()

        variance = np.asarray(
            hdul["VARIANCE"].data, dtype=np.float32
        ).copy()

        # IMAGE header: celestial coordinates and metadata
        header = hdul["IMAGE"].header.copy()

        # NEW: FLAGS header contains bitmask definitions
        flags_header = hdul["FLAGS"].header.copy()

        wcs = WCS(header, key=" ").celestial

        wave_data = hdul["WCS-WAVE"].data[0]

        wavelength_grid = np.array(
            wave_data["VALUES"], copy=True
        )

        wave_x = np.array(
            wave_data["X"], copy=True
        )

        wave_y = np.array(
            wave_data["Y"], copy=True
        )

        if not (
            image.shape == flags.shape == variance.shape
        ):
            raise ValueError(
                "IMAGE, FLAGS and VARIANCE shapes differ."
            )

        metadata = {
            "filename": filepath.name,
            "date_obs": header.get("DATE-OBS"),
            "unit": header.get("BUNIT"),
            "shape": image.shape,
        }

    return {
        "image": image,
        "flags": flags,
        "flags_header": flags_header,  # NEW
        "variance": variance,
        "wcs": wcs,
        "wavelength_grid": wavelength_grid,
        "wave_x": wave_x,
        "wave_y": wave_y,
        "metadata": metadata,
    }


if __name__ == "__main__":

    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / "data" / "raw"

    files = sorted(raw_dir.glob("*.fits"))

    if not files:
        raise FileNotFoundError(
            f"No FITS files found in {raw_dir}"
        )

    data = load_spherex_fits(files[0])

    print("\nSPHEREx FITS loaded successfully!")
    print("--------------------------------")

    for key, value in data["metadata"].items():
        print(f"{key}: {value}")

    print("Image shape:", data["image"].shape)
    print("Flags shape:", data["flags"].shape)
    print("Variance shape:", data["variance"].shape)

    print(
        "Wavelength grid shape:",
        data["wavelength_grid"].shape
    )

    print(
        "Celestial WCS dimensions:",
        data["wcs"].pixel_n_dim
    )

    # Verify that flag definitions were loaded
    print("\nFlag definitions loaded:")

    for key in data["flags_header"]:
        if key.startswith("MSKN"):
            print(
                key,
                "=",
                data["flags_header"][key]
            )
