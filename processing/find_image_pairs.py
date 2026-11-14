
from pathlib import Path
from astropy.table import Table
from astropy.time import Time
from astropy.coordinates import SkyCoord
import astropy.units as u

BASE_DIR = Path(__file__).resolve().parent.parent
csv_path = BASE_DIR / "spherex_observations.csv"

observations = Table.read(csv_path, format="csv")

dates = Time(
    observations["t_min"],
    format="mjd"
).to_datetime()

coords = SkyCoord(
    ra=observations["s_ra"] * u.deg,
    dec=observations["s_dec"] * u.deg
)

# Preliminary maximum center separation
max_separation = 0.5 * u.deg

pairs = []

for i in range(len(observations)):

    for j in range(i + 1, len(observations)):

        # Same spectral group
        if (
            observations["energy_bandpassname"][i]
            != observations["energy_bandpassname"][j]
        ):
            continue

        # Different observation months
        if dates[i].strftime("%Y-%m") == dates[j].strftime("%Y-%m"):
            continue

        separation = coords[i].separation(coords[j])

        if separation < max_separation:

            time_difference = abs(
                (dates[j] - dates[i]).days
            )

            if time_difference < 30:
                continue

            pairs.append({
                "index_1": i,
                "index_2": j,
                "date_1": dates[i].strftime("%Y-%m-%d"),
                "date_2": dates[j].strftime("%Y-%m-%d"),
                "band": observations["energy_bandpassname"][i],
                "separation_deg": separation.deg,
                "days_apart": time_difference,
                "url_1": observations["access_url"][i],
                "url_2": observations["access_url"][j]
            })

# Prioritize long time baseline and close centers
pairs.sort(
    key=lambda p: (
        -p["days_apart"],
        p["separation_deg"]
    )
)

print(f"Candidate pairs found: {len(pairs)}")

print("\nTop 10 candidate pairs:\n")

for pair in pairs[:10]:

    print("------------------------------")
    print("Indices:", pair["index_1"], pair["index_2"])
    print("Date 1:", pair["date_1"])
    print("Date 2:", pair["date_2"])
    print("Band:", pair["band"])
    print("Separation:", round(pair["separation_deg"], 4), "deg")
    print("Days apart:", pair["days_apart"])

# Save all candidate pairs
if pairs:
    from astropy.table import Table as AstroTable

    output = BASE_DIR / "data" / "candidate_pairs.csv"
    output.parent.mkdir(parents=True, exist_ok=True)

    AstroTable(rows=pairs).write(
        output,
        format="csv",
        overwrite=True
    )

    print("\nSaved candidate pairs to:", output)
