import pandas as pd

FILES = {
    "trawlers": "data/processed/gfw_trawlers_10min.parquet",
    "purse_seines": "data/processed/gfw_purse_seines_10min.parquet",
    "fixed_gear": "data/processed/gfw_fixed_gear_10min.parquet",
    "pole_and_line": "data/processed/gfw_pole_and_line_10min.parquet",
    "trollers": "data/processed/gfw_trollers_10min.parquet",
}

datasets = []

for vessel_type, file in FILES.items():
    print(f"Loading {vessel_type}...")

    df = pd.read_parquet(file)
    df["vessel_type"] = vessel_type

    datasets.append(df)

combined = pd.concat(datasets, ignore_index=True)

print("\n========== COMBINED DATASET ==========")
print("Total rows:", len(combined))
print("Total vessel IDs:", combined["mmsi"].nunique())
print("Total segments:", combined[["mmsi", "segment_id", "vessel_type"]].drop_duplicates().shape[0])

print("\n========== VESSELS BY CATEGORY ==========")
print(
    combined.groupby("vessel_type")["mmsi"]
    .nunique()
    .sort_values(ascending=False)
)

print("\n========== ROWS BY CATEGORY ==========")
print(combined["vessel_type"].value_counts())

print("\n========== MISSING VALUES ==========")
print(combined.isna().sum())

print("\n========== TIME RANGE ==========")
print("Start:", combined["timestamp"].min())
print("End:", combined["timestamp"].max())

print("\n========== GEOGRAPHIC COVERAGE ==========")
print("Latitude:")
print("  Min:", combined["lat"].min())
print("  Max:", combined["lat"].max())

print("Longitude:")
print("  Min:", combined["lon"].min())
print("  Max:", combined["lon"].max())

print("\n========== SPEED ==========")
print(combined["speed"].describe())

print("\n========== COURSE ==========")
print(combined["course"].describe())

print("\n========== SEGMENT LENGTH ==========")

segment_lengths = (
    combined
    .groupby(["vessel_type", "mmsi", "segment_id"])
    .size()
)

print(segment_lengths.describe())

print("\nSegments >= 30 observations:")
print((segment_lengths >= 30).sum())

print("\nSegments >= 50 observations:")
print((segment_lengths >= 50).sum())

print("\nSegments >= 100 observations:")
print((segment_lengths >= 100).sum())

print("\nSegments >= 200 observations:")
print((segment_lengths >= 200).sum())