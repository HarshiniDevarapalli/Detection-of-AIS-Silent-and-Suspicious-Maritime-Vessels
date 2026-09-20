import sys
from pathlib import Path

import numpy as np
import pandas as pd


MAX_GAP_MINUTES = 30


def haversine_km(lat1, lon1, lat2, lon2):
    """
    Calculate great-circle distance between two coordinates.
    """
    R = 6371.0

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    dlat = lat2 - lat1
    dlon = np.radians(lon2 - lon1)

    # Handle longitude wrap-around
    dlon = (dlon + np.pi) % (2 * np.pi) - np.pi

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def analyze_dataset(dataset_name):

    path = Path(f"data/processed/gfw_{dataset_name}_10min.parquet")

    if not path.exists():
        print(f"File not found: {path}")
        return

    print("=" * 70)
    print(f"DATASET: {dataset_name}")
    print("=" * 70)

    df = pd.read_parquet(path)

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Make sure data is ordered correctly
    df = df.sort_values(["mmsi", "segment_id", "timestamp"]).reset_index(drop=True)

    # ---------------------------------------------------------
    # Previous observation
    # ---------------------------------------------------------

    df["prev_lat"] = df.groupby(["mmsi", "segment_id"])["lat"].shift(1)
    df["prev_lon"] = df.groupby(["mmsi", "segment_id"])["lon"].shift(1)
    df["prev_timestamp"] = df.groupby(
        ["mmsi", "segment_id"]
    )["timestamp"].shift(1)

    # ---------------------------------------------------------
    # Time difference
    # ---------------------------------------------------------

    df["time_gap_minutes"] = (
        df["timestamp"] - df["prev_timestamp"]
    ).dt.total_seconds() / 60

    # ---------------------------------------------------------
    # Distance between consecutive observations
    # ---------------------------------------------------------

    valid_coordinates = (
        df["prev_lat"].notna()
        & df["prev_lon"].notna()
        & df["lat"].notna()
        & df["lon"].notna()
    )

    df["distance_km"] = np.nan

    df.loc[valid_coordinates, "distance_km"] = haversine_km(
        df.loc[valid_coordinates, "prev_lat"],
        df.loc[valid_coordinates, "prev_lon"],
        df.loc[valid_coordinates, "lat"],
        df.loc[valid_coordinates, "lon"],
    )

    # ---------------------------------------------------------
    # Implied speed
    #
    # km / hour
    # converted to knots
    # ---------------------------------------------------------

    df["implied_speed_knots"] = (
        df["distance_km"]
        / (df["time_gap_minutes"] / 60)
        / 1.852
    )

    # ---------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------

    print(f"Rows:       {len(df):,}")
    print(f"Vessels:    {df['mmsi'].nunique():,}")
    print(f"Segments:   {df['segment_id'].nunique():,}")

    print("\n--- Reported speed ---")

    print(
        f"Maximum reported speed: "
        f"{df['speed'].max():.2f} knots"
    )

    print(
        f"Speed = 102.3: "
        f"{(df['speed'] == 102.3).sum():,}"
    )

    print(
        f"Speed > 40: "
        f"{(df['speed'] > 40).sum():,}"
    )

    print("\n--- Implied movement speed ---")

    valid_speed = df["implied_speed_knots"].dropna()

    print(
        f"Median: {valid_speed.median():.2f} knots"
    )

    print(
        f"95th percentile: "
        f"{valid_speed.quantile(0.95):.2f} knots"
    )

    print(
        f"99th percentile: "
        f"{valid_speed.quantile(0.99):.2f} knots"
    )

    print(
        f"Maximum: "
        f"{valid_speed.max():.2f} knots"
    )

    print(
        f"Implied speed > 40 knots: "
        f"{(df['implied_speed_knots'] > 40).sum():,}"
    )

    print(
        f"Implied speed > 100 knots: "
        f"{(df['implied_speed_knots'] > 100).sum():,}"
    )

    # ---------------------------------------------------------
    # Show largest movement anomalies
    # ---------------------------------------------------------

    print("\n--- Top 20 largest implied-speed observations ---")

    cols = [
        "mmsi",
        "segment_id",
        "timestamp",
        "lat",
        "lon",
        "speed",
        "time_gap_minutes",
        "distance_km",
        "implied_speed_knots",
    ]

    top = (
        df.loc[df["implied_speed_knots"].notna(), cols]
        .sort_values("implied_speed_knots", ascending=False)
        .head(20)
    )

    print(top.to_string(index=False))

    # ---------------------------------------------------------
    # Reported vs implied speed difference
    # ---------------------------------------------------------

    df["speed_difference"] = (
        df["implied_speed_knots"] - df["speed"]
    ).abs()

    print("\n--- Largest reported/implied speed differences ---")

    diff = (
        df.loc[df["speed_difference"].notna(), cols + ["speed_difference"]]
        .sort_values("speed_difference", ascending=False)
        .head(20)
    )

    print(diff.to_string(index=False))


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print(
            "Usage: python scripts/validate_movement.py "
            "<dataset_name>"
        )
        sys.exit(1)

    analyze_dataset(sys.argv[1])