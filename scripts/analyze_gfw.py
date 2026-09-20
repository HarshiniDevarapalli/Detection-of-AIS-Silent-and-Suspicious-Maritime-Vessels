import pandas as pd

FILE = "data/processed/gfw_trawlers_clean.csv"

df = pd.read_csv(FILE)

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

# Sort by vessel and time
df = df.sort_values(["mmsi", "timestamp"])

# Time difference between consecutive observations
df["time_diff"] = df.groupby("mmsi")["timestamp"].diff()

print("========== DATASET ==========")
print("Rows:", len(df))
print("Unique vessels:", df["mmsi"].nunique())

print("\n========== TIME GAPS ==========")
print(df["time_diff"].describe())

print("\n========== COMMON TIME GAPS ==========")
print(
    df["time_diff"]
    .value_counts()
    .head(20)
)

print("\n========== VESSEL OBSERVATIONS ==========")
observations_per_vessel = df.groupby("mmsi").size()

print(observations_per_vessel.describe())

print("\n========== TOP VESSELS ==========")
print(
    observations_per_vessel
    .sort_values(ascending=False)
    .head(20)
)