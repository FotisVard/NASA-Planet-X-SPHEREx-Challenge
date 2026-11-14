
from pathlib import Path
from urllib.parse import urlparse
import requests
import numpy as np
import matplotlib.pyplot as plt

from astropy.table import Table
from astropy.io import fits
from astropy.visualization import ImageNormalize, ZScaleInterval, AsinhStretch

# -----------------------------------
# 1. Project paths
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = BASE_DIR / "spherex_observations.csv"
DATA_DIR = BASE_DIR / "data" / "raw"

DATA_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------
# 2. Read SPHEREx observations
# -----------------------------------

observations = Table.read(CSV_PATH, format="csv")

print(f"Total observations: {len(observations)}")

# Select first FITS URL
fits_url = str(observations["access_url"][0])

filename = Path(urlparse(fits_url).path).name
fits_path = DATA_DIR / filename

print(f"Selected file: {filename}")

# -----------------------------------
# 3. Download FITS file
# -----------------------------------

if not fits_path.exists():

    print("Downloading FITS file...")

    with requests.get(
        fits_url,
        stream=True,
        timeout=(20, 120)
    ) as response:

        response.raise_for_status()

        with open(fits_path, "wb") as file:
            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):
                if chunk:
                    file.write(chunk)

    print("Download completed!")

else:
    print("File already downloaded.")

# -----------------------------------
# 4. Inspect FITS structure
# -----------------------------------

with fits.open(fits_path, memmap=True) as hdul:

    hdul.info()

    image_data = None

    for hdu in hdul:
        if hdu.data is None:
            continue

        data = np.asarray(hdu.data)

        if data.ndim == 2 and np.issubdtype(
            data.dtype, np.number
        ):
            image_data = np.array(data, dtype=float)
            print(f"Image extension: {hdu.name}")
            break

    if image_data is None:
        raise ValueError(
            "No 2D image found in this FITS file."
        )

# -----------------------------------
# 5. Visualize image
# -----------------------------------

valid_pixels = np.isfinite(image_data)

if not np.any(valid_pixels):
    raise ValueError("Image has no finite pixels.")

norm = ImageNormalize(
    image_data[valid_pixels],
    interval=ZScaleInterval(),
    stretch=AsinhStretch()
)

plt.figure(figsize=(9, 7))

plt.imshow(
    image_data,
    origin="lower",
    cmap="gray",
    norm=norm
)

plt.colorbar(label="Pixel value")
plt.title("NASA SPHEREx - First FITS Image")
plt.xlabel("Pixel X")
plt.ylabel("Pixel Y")
plt.tight_layout()

output_path = BASE_DIR / "data" / "first_spherex_image.png"

plt.savefig(output_path, dpi=150)

print(f"Image saved: {output_path}")

plt.show()
