import sys
import pandas as pd


# --------------------------------------------------
# 1. Get dataset name
# --------------------------------------------------

if len(sys.argv) != 2:
    print("Usage: python scripts/analyze_sampling.py <dataset_name>")
    print("Example: python scripts/analyze_sampling.py purse_seines")
    sys.exit(1)

DATASET = sys.argv[1]

INPUT_FILE = f"data/processed/{DATASET}_clean.csv"


# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

print(f"Loading: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)


# --------------------------------------------------
# 3. Convert timestamp
# --------------------------------------------------

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True
)


# --------------------------------------------------
# 4. Sort by vessel and time
# --------------------------------------------------

df = df.sort_values(
    ["mmsi", "timestamp"]
).reset_index(drop=True)


# --------------------------------------------------
# 5. Calculate time gaps
# --------------------------------------------------

df["time_diff"] = (
    df.groupby("mmsi")["timestamp"].diff()
)


# --------------------------------------------------
# 6. Create trajectory segments
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
# 7. Find usable segments
# --------------------------------------------------

segment_sizes = (
    df.groupby(["mmsi", "segment_id"])
      .size()
      .reset_index(name="observations")
)

usable_segments = segment_sizes[
    segment_sizes["observations"] >= 100
]


# --------------------------------------------------
# 8. Keep only usable segments
# --------------------------------------------------

df = df.merge(
    usable_segments[
        ["mmsi", "segment_id"]
    ],
    on=["mmsi", "segment_id"],
    how="inner"
)


# --------------------------------------------------
# 9. Analyze sampling gaps
# --------------------------------------------------

gaps = df["time_diff"].dropna()

gaps = gaps[
    gaps <= MAX_GAP
]


# --------------------------------------------------
# 10. Dataset summary
# --------------------------------------------------

print("\n========== DATASET ==========")

print("Dataset:", DATASET)

print("Observations:", len(df))

print(
    "Vessels:",
    df["mmsi"].nunique()
)

print(
    "Usable segments:",
    len(usable_segments)
)


# --------------------------------------------------
# 11. Sampling gap statistics
# --------------------------------------------------

print("\n========== SAMPLING GAP ==========")

print(gaps.describe())


# --------------------------------------------------
# 12. Gap quantiles
# --------------------------------------------------

print("\n========== GAP QUANTILES ==========")

print("25% :", gaps.quantile(0.25))
print("50% :", gaps.quantile(0.50))
print("75% :", gaps.quantile(0.75))
print("90% :", gaps.quantile(0.90))
print("95% :", gaps.quantile(0.95))
print("99% :", gaps.quantile(0.99))


# --------------------------------------------------
# 13. Gap distribution
# --------------------------------------------------

print("\n========== GAP DISTRIBUTION ==========")

for minutes in [1, 2, 5, 10, 15, 20, 30]:

    threshold = pd.Timedelta(
        minutes=minutes
    )

    percentage = (
        gaps <= threshold
    ).mean() * 100

    print(
        f"Gaps <= {minutes:2d} minutes: "
        f"{percentage:.2f}%"
    )