import pandas as pd

INPUT_FILE = "data/processed/gfw_trawlers_clean.csv"
OUTPUT_FILE = "data/processed/gfw_trawlers_10min.parquet"

# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

df = df.sort_values(
    ["mmsi", "timestamp"]
).reset_index(drop=True)

# --------------------------------------------------
# 2. Calculate time gaps
# --------------------------------------------------

df["time_diff"] = (
    df.groupby("mmsi")["timestamp"].diff()
)

# --------------------------------------------------
# 3. Create trajectory segments
# --------------------------------------------------

MAX_GAP = pd.Timedelta(minutes=30)

df["new_segment"] = (
    df["time_diff"].isna()
    | (df["time_diff"] > MAX_GAP)
)

df["segment_id"] = (
    df.groupby("mmsi")["new_segment"]
      .cumsum()
)

# --------------------------------------------------
# 4. Keep only sufficiently long segments
# --------------------------------------------------

segment_sizes = (
    df.groupby(["mmsi", "segment_id"])
      .size()
      .reset_index(name="observations")
)

usable_segments = segment_sizes[
    segment_sizes["observations"] >= 100
]

print("Total usable segments:", len(usable_segments))

df = df.merge(
    usable_segments[["mmsi", "segment_id"]],
    on=["mmsi", "segment_id"],
    how="inner"
)

print("Observations before resampling:", len(df))

# --------------------------------------------------
# 5. Resample each trajectory to 10-minute intervals
# --------------------------------------------------

features = [
    "lat",
    "lon",
    "speed",
    "course"
]

resampled_segments = []

for (mmsi, segment_id), group in df.groupby(
    ["mmsi", "segment_id"]
):

    group = group.sort_values("timestamp")

    group = group.set_index("timestamp")

    # Resample to 10-minute intervals
    resampled = group[features].resample("10min").mean()

    # Interpolate only small gaps
    resampled = resampled.interpolate(
        method="time",
        limit=1
    )

    # Remove rows where required features are still missing
    resampled = resampled.dropna(
        subset=["lat", "lon", "speed", "course"]
    )

    # Restore identifiers
    resampled["mmsi"] = mmsi
    resampled["segment_id"] = segment_id

    resampled_segments.append(
        resampled.reset_index()
    )

# --------------------------------------------------
# 6. Combine everything
# --------------------------------------------------

result = pd.concat(
    resampled_segments,
    ignore_index=True
)

result = result[
    [
        "mmsi",
        "segment_id",
        "timestamp",
        "lat",
        "lon",
        "speed",
        "course"
    ]
]

# --------------------------------------------------
# 7. Save as Parquet
# --------------------------------------------------

result.to_parquet(
    OUTPUT_FILE,
    index=False
)

# --------------------------------------------------
# 8. Print summary
# --------------------------------------------------

print("\n========== RESAMPLED DATA ==========")

print("Rows:", len(result))

print(
    "Vessels:",
    result["mmsi"].nunique()
)

print(
    "Segments:",
    result[["mmsi", "segment_id"]]
    .drop_duplicates()
    .shape[0]
)

print("\nFirst 10 rows:")
print(result.head(10))

print(
    f"\nSaved to: {OUTPUT_FILE}"
)