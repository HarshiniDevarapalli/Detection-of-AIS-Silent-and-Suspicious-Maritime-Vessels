import pandas as pd

INPUT_FILE = "data/processed/gfw_trawlers_clean.csv"

df = pd.read_csv(INPUT_FILE)

df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

df = df.sort_values(["mmsi", "timestamp"]).reset_index(drop=True)

# Calculate time differences within each vessel
df["time_diff"] = df.groupby("mmsi")["timestamp"].diff()

MAX_GAP = pd.Timedelta(minutes=30)

df["new_segment"] = (
    df["time_diff"].isna() |
    (df["time_diff"] > MAX_GAP)
)

df["segment_id"] = (
    df.groupby("mmsi")["new_segment"]
      .cumsum()
)

# Group by vessel + segment
segments = (
    df.groupby(["mmsi", "segment_id"])
      .agg(
          observations=("timestamp", "size"),
          start_time=("timestamp", "min"),
          end_time=("timestamp", "max")
      )
      .reset_index()
)

# Calculate duration
segments["duration"] = (
    segments["end_time"] - segments["start_time"]
)

# Only look at reasonably long trajectories
usable = segments[segments["observations"] >= 100].copy()

print("========== ALL SEGMENTS ==========")
print("Total segments:", len(segments))

print("\n========== USABLE SEGMENTS ==========")
print("Segments >= 100 observations:", len(usable))

print("\n========== OBSERVATIONS ==========")
print(usable["observations"].describe())

print("\n========== DURATION ==========")
print(usable["duration"].describe())

print("\n========== TOP 20 LONGEST ==========")
print(
    usable.sort_values("duration", ascending=False)
          .head(20)
          [["mmsi", "segment_id", "observations",
            "start_time", "end_time", "duration"]]
)