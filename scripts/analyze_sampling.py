import pandas as pd

INPUT_FILE = "data/processed/gfw_trawlers_clean.csv"

df = pd.read_csv(INPUT_FILE)

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

# Sort by vessel and time
df = df.sort_values(["mmsi", "timestamp"]).reset_index(drop=True)

# Calculate time gap between consecutive observations
df["time_diff"] = df.groupby("mmsi")["timestamp"].diff()

# Same segmentation rule we already used
MAX_GAP = pd.Timedelta(minutes=30)

df["new_segment"] = (
    df["time_diff"].isna() |
    (df["time_diff"] > MAX_GAP)
)

df["segment_id"] = (
    df.groupby("mmsi")["new_segment"]
      .cumsum()
)

# Find segments with at least 100 observations
segment_sizes = (
    df.groupby(["mmsi", "segment_id"])
      .size()
      .reset_index(name="observations")
)

usable_segments = segment_sizes[
    segment_sizes["observations"] >= 100
]

# Keep only observations belonging to usable segments
df = df.merge(
    usable_segments[["mmsi", "segment_id"]],
    on=["mmsi", "segment_id"],
    how="inner"
)

# Remove gaps larger than 30 minutes
gaps = df["time_diff"].dropna()
gaps = gaps[gaps <= MAX_GAP]

print("========== USABLE DATA ==========")
print("Observations:", len(df))
print("Vessels:", df["mmsi"].nunique())
print("Usable segments:", len(usable_segments))

print("\n========== SAMPLING GAP ==========")
print(gaps.describe())

print("\n========== GAP QUANTILES ==========")
print("25% :", gaps.quantile(0.25))
print("50% :", gaps.quantile(0.50))
print("75% :", gaps.quantile(0.75))
print("90% :", gaps.quantile(0.90))
print("95% :", gaps.quantile(0.95))
print("99% :", gaps.quantile(0.99))

print("\n========== GAP DISTRIBUTION ==========")

for minutes in [1, 2, 5, 10, 15, 20, 30]:

    threshold = pd.Timedelta(minutes=minutes)

    percentage = (gaps <= threshold).mean() * 100

    print(
        f"Gaps <= {minutes:2d} minutes: "
        f"{percentage:.2f}%"
    )