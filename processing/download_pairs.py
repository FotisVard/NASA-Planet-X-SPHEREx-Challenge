
import argparse
from pathlib import Path
from urllib.parse import urlparse

import requests
from astropy.table import Table


BASE_DIR = Path(__file__).resolve().parent.parent

PAIRS_FILE = (
    BASE_DIR / "data" / "searches" / "best_pairs.csv"
)

RAW_DIR = BASE_DIR / "data" / "raw"


def download_fits(url, output_dir):
    """
    Download a FITS file with streaming.
    Skip existing non-empty files.
    """

    filename = Path(urlparse(url).path).name

    if not filename.endswith(".fits"):
        raise ValueError(f"Unexpected FITS URL: {url}")

    output_path = output_dir / filename

    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Already exists: {filename}")
        return output_path

    temp_path = output_path.with_suffix(".fits.part")

    print(f"\nDownloading: {filename}")

    try:
        with requests.get(
            url,
            stream=True,
            timeout=(30, 180)
        ) as response:

            response.raise_for_status()

            with open(temp_path, "wb") as file:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        file.write(chunk)

        if temp_path.stat().st_size == 0:
            raise RuntimeError("Downloaded file is empty.")

        temp_path.replace(output_path)

    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    print(f"Saved: {output_path}")

    return output_path


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--max-pairs",
        type=int,
        default=1,
        help="Number of pairs to download"
    )

    args = parser.parse_args()

    if args.max_pairs < 1:
        parser.error("--max-pairs must be >= 1")

    pairs = Table.read(
        PAIRS_FILE,
        format="csv"
    )

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    selected = pairs[:args.max_pairs]

    downloaded = set()

    for pair in selected:

        print(
            f"\nDetector: {pair['detector']}"
        )

        for column in ["url_1", "url_2"]:

            url = str(pair[column])

            if url in downloaded:
                continue

            download_fits(url, RAW_DIR)

            downloaded.add(url)

    print("\nDownload process completed.")


if __name__ == "__main__":
    main()
