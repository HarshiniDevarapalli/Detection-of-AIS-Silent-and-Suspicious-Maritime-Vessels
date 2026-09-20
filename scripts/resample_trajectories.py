import sys
import os
import pandas as pd


# --------------------------------------------------
# 1. Get dataset name
# --------------------------------------------------

if len(sys.argv) != 2:
    print("Usage: python scripts/resample_trajectories.py <dataset_name>")
    print("Example: python scripts/resample_trajectories.py purse_seines")
    sys.exit(1)

DATASET = sys.argv[1]

INPUT_FILE = f"data/processed/{DATASET}_clean.csv"
OUTPUT_FILE = f"data/processed/gfw_{DATASET}_10min.parquet"


# --------------------------------------------------
# 2. Check input file
# --------------------------------------------------

if not os.path.exists(INPUT_FILE):
    print(f"Error: File not found: {INPUT_FILE}")
    sys.exit(1)


# --------------------------------------------------
# 3. Load data
# --------------------------------------------------

print(f"Loading: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True
)

df = df.sort_values(
    ["mmsi", "timestamp"]
).reset_index(drop=True)


# --------------------------------------------------
# 4. Calculate time gaps
# --------------------------------------------------

df["time_diff"] = (
    df.groupby("mmsi")["timestamp"].diff()
)


# --------------------------------------------------
# 5. Create trajectory segments
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
# 6. Keep sufficiently long segments
# --------------------------------------------------

segment_sizes = (
    df.groupby(["mmsi", "segment_id"])
      .size()
      .reset_index(name="observations")
)

usable_segments = segment_sizes[
    segment_sizes["observations"] >= 100
]

print("\n========== SEGMENTS ==========")

print(
    "Total usable segments:",
    len(usable_segments)
)

df = df.merge(
    usable_segments[
        ["mmsi", "segment_id"]
    ],
    on=["mmsi", "segment_id"],
    how="inner"
)

print(
    "Observations before resampling:",
    len(df)
)


# --------------------------------------------------
# 7. Resample trajectories
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
    resampled = (
        group[features]
        .resample("10min")
        .mean()
    )

    # Interpolate only one missing timestep
    resampled = resampled.interpolate(
        method="time",
        limit=1
    )

    # Remove remaining missing values
    resampled = resampled.dropna(
        subset=features
    )

    # Restore identifiers
    resampled["mmsi"] = mmsi
    resampled["segment_id"] = segment_id

    resampled_segments.append(
        resampled.reset_index()
    )


# --------------------------------------------------
# 8. Combine all segments
# --------------------------------------------------

result = pd.concat(
    resampled_segments,
    ignore_index=True
)


# --------------------------------------------------
# 9. Select final columns
# --------------------------------------------------

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
# 10. Save as Parquet
# --------------------------------------------------

result.to_parquet(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# 11. Print summary
# --------------------------------------------------

print("\n========== RESAMPLED DATA ==========")

print("Dataset:", DATASET)

print(
    "Rows:",
    len(result)
)

print(
    "Vessels:",
    result["mmsi"].nunique()
)

print(
    "Segments:",
    result[
        ["mmsi", "segment_id"]
    ].drop_duplicates().shape[0]
)

print("\nFirst 10 rows:")

print(result.head(10))

print(
    f"\nSaved to: {OUTPUT_FILE}"
)