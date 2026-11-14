
from pathlib import Path
from urllib.parse import urlparse

import requests
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.visualization import (
    ImageNormalize,
    ZScaleInterval,
    AsinhStretch
)
from reproject import reproject_interp

# -----------------------------------
# 1. Project paths
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------
# 2. Read candidate pair
# -----------------------------------

pairs = Table.read(
    BASE_DIR / "data" / "candidate_pairs.csv",
    format="csv"
)

pair = pairs[0]

urls = [
    str(pair["url_1"]),
    str(pair["url_2"])
]

# -----------------------------------
# 3. Download FITS files
# -----------------------------------

def download_fits(url):

    filename = Path(urlparse(url).path).name
    filepath = DATA_DIR / filename

    if filepath.exists():
        print(f"Already downloaded: {filename}")
        return filepath

    print(f"Downloading: {filename}")

    try:
        with requests.get(
            url,
            stream=True,
            timeout=(20, 180)
        ) as response:

            response.raise_for_status()

            with open(filepath, "wb") as f:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        f.write(chunk)

    except Exception:
        filepath.unlink(missing_ok=True)
        raise

    print("Download completed.")
    return filepath


paths = [download_fits(url) for url in urls]

# -----------------------------------
# 4. Load scientific images
# -----------------------------------

def load_image(path):

    with fits.open(path, memmap=False) as hdul:

        image = hdul["IMAGE"].data.astype(float)
        header = hdul["IMAGE"].header.copy()

        wcs = WCS(header).celestial

        print(f"\nFile: {path.name}")
        print(f"Shape: {image.shape}")
        print(f"BUNIT: {header.get('BUNIT', 'Unknown')}")
        print(f"WCS axes: {wcs.pixel_n_dim}")

        return image, wcs


image1, wcs1 = load_image(paths[0])
image2, wcs2 = load_image(paths[1])

# -----------------------------------
# 5. Reproject second image
# -----------------------------------

print("\nReprojecting second image...")

image2_aligned, footprint = reproject_interp(
    (image2, wcs2),
    wcs1,
    shape_out=image1.shape
)

valid = (
    np.isfinite(image1)
    & np.isfinite(image2_aligned)
    & (footprint > 0)
)

overlap = np.sum(valid) / valid.size * 100

print(f"Valid overlap: {overlap:.2f}%")

if not np.any(valid):
    raise ValueError("No valid overlap between images.")

# -----------------------------------
# 6. Difference image
# -----------------------------------

difference = np.full_like(image1, np.nan)

difference[valid] = (
    image2_aligned[valid] - image1[valid]
)

# -----------------------------------
# 7. Visualization
# -----------------------------------

norm = ImageNormalize(
    image1[valid],
    interval=ZScaleInterval(),
    stretch=AsinhStretch()
)

fig, axes = plt.subplots(
    1, 3,
    figsize=(18, 6)
)

axes[0].imshow(
    image1,
    origin="lower",
    cmap="gray",
    norm=norm
)

axes[0].set_title("SPHEREx - May 2025")

axes[1].imshow(
    image2_aligned,
    origin="lower",
    cmap="gray",
    norm=norm
)

axes[1].set_title("SPHEREx - May 2026")

diff_values = difference[valid]

limit = np.nanpercentile(
    np.abs(diff_values),
    99
)

if not np.isfinite(limit) or limit == 0:
    limit = 1

axes[2].imshow(
    difference,
    origin="lower",
    cmap="RdBu_r",
    vmin=-limit,
    vmax=limit
)

axes[2].set_title("Difference Image")

for ax in axes:
    ax.set_xlabel("Pixel X")
    ax.set_ylabel("Pixel Y")

plt.tight_layout()

output_path = OUTPUT_DIR / "comparison.png"

plt.savefig(output_path, dpi=150)
print(f"\nComparison saved: {output_path}")

plt.show()
