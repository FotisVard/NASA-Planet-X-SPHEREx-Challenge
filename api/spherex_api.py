
from astroquery.ipac.irsa import Irsa
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.utils.data import conf

# Increase timeout for archive requests
conf.remote_timeout = 120

# ---------------------------------
# 1. Target object: WISE 0855
# ---------------------------------

object_name = "WISE 0855"

ra = 133.795
dec = -7.245

coord = SkyCoord(
    ra=ra,
    dec=dec,
    unit="deg",
    frame="icrs"
)

# Search radius: 1 arcminute
search_radius = 1 * u.arcmin

# ---------------------------------
# 2. SPHEREx data collections
# ---------------------------------

collections = [
    "spherex_dr1",
    "spherex_qr3"
]

# ---------------------------------
# 3. Query SPHEREx API
# ---------------------------------

print("Connecting to SPHEREx Archive...")
print(f"Object: {object_name}")
print(f"RA: {ra}")
print(f"DEC: {dec}")

try:
    results = Irsa.query_sia(
        pos=(coord, search_radius),
        collection=collections
    )

    # ---------------------------------
    # 4. Display results
    # ---------------------------------

    print("\nSearch completed!")

    number_of_results = len(results)

    print(f"Observations found: {number_of_results}")

    if number_of_results > 0:

        columns_to_display = [
            "obs_id",
            "s_ra",
            "s_dec",
            "access_url",
            "t_min",
            "t_max"
        ]

        available_columns = [
            col for col in columns_to_display
            if col in results.colnames
        ]

        print("\nAvailable observations:\n")
        print(results[available_columns][:10])

        # ---------------------------------
        # 5. Save results
        # ---------------------------------

        results.write(
            "spherex_observations.csv",
            format="csv",
            overwrite=True
        )

        print("\nResults saved to:")
        print("spherex_observations.csv")

    else:
        print("No observations found for this position.")

except Exception as error:
    print("\nAn error occurred:")
    print(error)