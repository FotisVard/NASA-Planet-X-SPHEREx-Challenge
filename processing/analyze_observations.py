
from pathlib import Path
from astropy.table import Table
from astropy.time import Time
from collections import Counter

BASE_DIR = Path(__file__).resolve().parent.parent
csv_path = BASE_DIR / "spherex_observations.csv"

observations = Table.read(csv_path, format="csv")

print("Total records:", len(observations))

# Count observations per spectral group
bands = Counter(observations["energy_bandpassname"])

print("\nObservations per band:")
for band, count in sorted(bands.items()):
    print(f"{band}: {count}")

# Convert MJD timestamps to calendar dates
dates = Time(
    observations["t_min"],
    format="mjd"
).to_datetime()

months = Counter(
    date.strftime("%Y-%m")
    for date in dates
)

print("\nObservations per month:")
for month, count in sorted(months.items()):
    print(f"{month}: {count}")

# Show first records with dates and bands
print("\nFirst 15 records:")

for i in range(min(15, len(observations))):
    print(
        i,
        dates[i].strftime("%Y-%m-%d"),
        observations["energy_bandpassname"][i],
        observations["obs_id"][i]
    )
