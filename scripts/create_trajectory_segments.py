import pandas as pd

INPUT_FILE = "data/processed/gfw_trawlers_clean.csv"

df = pd.read_csv(INPUT_FILE)

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

df = df.sort_values(["mmsi", "timestamp"]).reset_index(drop=True)

# Time difference within each vessel
df["time_diff"] = df.groupby("mmsi")["timestamp"].diff()

# Maximum gap allowed inside a continuous trajectory
MAX_GAP = pd.Timedelta(minutes=30)

df["new_segment"] = (
    df["time_diff"].isna() |
    (df["time_diff"] > MAX_GAP)
)

# Segment number within each vessel
df["segment_id"] = (
    df.groupby("mmsi")["new_segment"]
      .cumsum()
)

# Count segments correctly using BOTH vessel and segment ID
segment_sizes = (
    df.groupby(["mmsi", "segment_id"])
      .size()
)

print("========== DATASET ==========")
print("Total observations:", len(df))
print("Total vessels:", df["mmsi"].nunique())
print("Total trajectory segments:", len(segment_sizes))

print("\n========== SEGMENT SIZE ==========")
print(segment_sizes.describe())

for threshold in [50, 100, 200, 500, 1000]:
    print(
        f"Segments >= {threshold}:",
        (segment_sizes >= threshold).sum()
    )