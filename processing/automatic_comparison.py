
import argparse
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

from fits_loader import load_spherex_fits
from image_comparison import compare_images


BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PAIRS_FILE = BASE_DIR / "data" / "searches" / "spectral_ranked_pairs.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "automatic"


def filename_from_url(url):
    return Path(urlparse(str(url)).path).name


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--detector",
        default="D1",
        help="Detector name, e.g. D1 or D5"
    )

    args = parser.parse_args()

    detector = args.detector.upper()

    if not detector.startswith("SPHEREX-"):
        detector = f"SPHEREx-{detector}"

    pairs = Table.read(PAIRS_FILE, format="csv")

    matches = [
        row for row in pairs
        if (
            str(row["detector"]).lower() == detector.lower()
            and str(row["within_tolerance"]).strip().lower()
            in ("true", "1")
        )
    ]

    if not matches:
        raise ValueError(f"No pair found for {detector}")

    pair = matches[0]

    path1 = RAW_DIR / filename_from_url(pair["url_1"])
    path2 = RAW_DIR / filename_from_url(pair["url_2"])

    for path in (path1, path2):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing FITS: {path}\n"
                "Download it before running this script."
            )

    print(f"\nDetector: {detector}")
    print("Image 1:", path1.name)
    print("Image 2:", path2.name)

    print("\nLoading FITS files...")

    data1 = load_spherex_fits(path1)
    data2 = load_spherex_fits(path2)

    print("\nAligning and comparing images...")

    result = compare_images(data1, data2)

    difference = np.asarray(
        result["difference"],
        dtype=np.float32
    )

    valid_mask = (
        np.asarray(result["valid_mask"], dtype=bool)
        & np.isfinite(difference)
    )

    valid_count = int(np.count_nonzero(valid_mask))
    total_count = int(difference.size)

    if valid_count == 0:
        raise RuntimeError("No valid overlapping pixels.")

    valid_values = difference[valid_mask]

    median = float(np.median(valid_values))
    mad_scatter = float(
        1.4826 * np.median(np.abs(valid_values - median))
    )

    print("\nComparison results:")
    print("Valid pixels:", valid_count)
    print(
        "Overlap:",
        round(100 * valid_count / total_count, 2),
        "%"
    )
    print("Median difference:", median, "MJy/sr")
    print("Robust scatter:", mad_scatter, "MJy/sr")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    limit = float(
        np.percentile(np.abs(valid_values - median), 99)
    )

    if not np.isfinite(limit) or limit <= 0:
        limit = 1.0

    display_image = np.where(
        valid_mask,
        difference - median,
        np.nan
    )

    fig, ax = plt.subplots(figsize=(9, 8))

    im = ax.imshow(
        display_image,
        origin="lower",
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit
    )

    ax.set_title(
        f"{detector}: 2026 - 2025\n"
        "WCS-aligned difference (median centered)"
    )

    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")

    fig.colorbar(
        im,
        ax=ax,
        label="Brightness difference (MJy/sr)"
    )

    output_image = OUTPUT_DIR / (
        f"{detector.replace('SPHEREx-', '')}_difference.png"
    )

    fig.savefig(
        output_image,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("\nSaved:", output_image)
    print("Automatic comparison completed.")


if __name__ == "__main__":
    main()
