
import argparse
from pathlib import Path

from astroquery.ipac.irsa import Irsa
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.utils.data import conf


# ---------------------------------------
# Configuration
# ---------------------------------------

conf.remote_timeout = 120

BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = BASE_DIR / "data" / "searches"

COLLECTIONS = [
    "spherex_dr1",
    "spherex_qr3"
]


# ---------------------------------------
# Search observations
# ---------------------------------------

def search_observations(ra, dec, radius):

    print("\nConnecting to SPHEREx Archive...")

    print(f"RA: {ra} degrees")
    print(f"DEC: {dec} degrees")
    print(f"Radius: {radius} arcmin")

    print(f"Collections: {COLLECTIONS}")

    coord = SkyCoord(
        ra=ra,
        dec=dec,
        unit="deg",
        frame="icrs"
    )

    search_radius = radius * u.arcmin

    results = Irsa.query_sia(
        pos=(coord, search_radius),
        collection=COLLECTIONS
    )

    print("\nSearch completed!")

    print(
        f"Observations found: {len(results)}"
    )

    return results


# ---------------------------------------
# Display observations
# ---------------------------------------

def display_observations(results):

    if len(results) == 0:
        print("\nNo observations found.")
        return

    columns_to_display = [
        "obs_id",
        "s_ra",
        "s_dec",
        "access_url",
        "t_min",
        "t_max",
        "em_min",
        "em_max"
    ]

    available_columns = [
        col
        for col in columns_to_display
        if col in results.colnames
    ]

    print("\nAvailable columns:")
    print(results.colnames)

    print("\nFirst 10 observations:\n")

    if available_columns:
        results[available_columns][:10].pprint(
            max_width=140,
            max_lines=12
        )
    else:
        results[:10].pprint(
            max_width=140,
            max_lines=12
        )


# ---------------------------------------
# Save results
# ---------------------------------------

def save_results(results, ra, dec, radius):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    filename = (
        f"spherex_ra{ra:.4f}_"
        f"dec{dec:.4f}_"
        f"r{radius:.2f}arcmin.csv"
    )

    output_path = OUTPUT_DIR / filename

    results.write(
        output_path,
        format="csv",
        overwrite=True
    )

    print("\nResults saved to:")
    print(output_path)

    return output_path


# ---------------------------------------
# Main
# ---------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Search SPHEREx observations "
            "for a selected sky region."
        )
    )

    parser.add_argument(
        "--ra",
        type=float,
        required=True,
        help="Right Ascension in degrees"
    )

    parser.add_argument(
        "--dec",
        type=float,
        required=True,
        help="Declination in degrees"
    )

    parser.add_argument(
        "--radius",
        type=float,
        default=1.0,
        help="Search radius in arcminutes"
    )

    args = parser.parse_args()

    if not 0 <= args.ra < 360:
        parser.error(
            "RA must be between 0 and 360 degrees."
        )

    if not -90 <= args.dec <= 90:
        parser.error(
            "DEC must be between -90 and 90 degrees."
        )

    if args.radius <= 0:
        parser.error(
            "Radius must be greater than zero."
        )

    try:

        results = search_observations(
            args.ra,
            args.dec,
            args.radius
        )

        if len(results) == 0:
            print(
                "\nNo SPHEREx observations "
                "found for this region."
            )
            return

        display_observations(results)

        save_results(
            results,
            args.ra,
            args.dec,
            args.radius
        )

        print(
            "\nSPHEREx region search "
            "completed successfully."
        )

    except Exception as error:

        print("\nSearch failed:")
        print(type(error).__name__)
        print(error)

        raise


if __name__ == "__main__":
    main()
