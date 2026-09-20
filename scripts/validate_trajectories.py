import pandas as pd
import numpy as np


FILES = {
    "trawlers": "data/processed/gfw_trawlers_10min.parquet",
    "purse_seines": "data/processed/gfw_purse_seines_10min.parquet",
    "fixed_gear": "data/processed/gfw_fixed_gear_10min.parquet",
    "pole_and_line": "data/processed/gfw_pole_and_line_10min.parquet",
    "trollers": "data/processed/gfw_trollers_10min.parquet",
}


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate great-circle distance between two geographic points.

    Returns:
        Distance in kilometers.
    """

    R = 6371.0

    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)

    dlat = lat2 - lat1

    # Handle longitude wraparound correctly.
    dlon = ((lon2 - lon1 + 180) % 360) - 180
    dlon = np.radians(dlon)

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1)
        * np.cos(lat2)
        * np.sin(dlon / 2) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


for vessel_type, file in FILES.items():

    print("\n" + "=" * 60)
    print(f"DATASET: {vessel_type}")
    print("=" * 60)

    # --------------------------------------------------
    # Load
    # --------------------------------------------------

    df = pd.read_parquet(file)

    print("\nRows:", len(df))
    print("Vessels:", df["mmsi"].nunique())

    segments = (
        df[["mmsi", "segment_id"]]
        .drop_duplicates()
        .shape[0]
    )

    print("Segments:", segments)

    # --------------------------------------------------
    # Missing values
    # --------------------------------------------------

    print("\n========== MISSING VALUES ==========")
    print(df.isna().sum())

    # --------------------------------------------------
    # Coordinates
    # --------------------------------------------------

    invalid_lat = (
        (df["lat"] < -90)
        | (df["lat"] > 90)
    )

    invalid_lon = (
        (df["lon"] < -180)
        | (df["lon"] > 180)
    )

    print("\n========== COORDINATES ==========")
    print("Invalid latitude:", invalid_lat.sum())
    print("Invalid longitude:", invalid_lon.sum())

    # --------------------------------------------------
    # Course
    # --------------------------------------------------

    invalid_course = (
        (df["course"] < 0)
        | (df["course"] > 360)
    )

    print("\n========== COURSE ==========")
    print("Invalid course values:", invalid_course.sum())
    print("Maximum course:", df["course"].max())

    if invalid_course.sum() > 0:

        print("\nExample invalid courses:")

        print(
            df.loc[
                invalid_course,
                [
                    "mmsi",
                    "segment_id",
                    "timestamp",
                    "course",
                ],
            ].head(10)
        )

    # --------------------------------------------------
    # Speed
    # --------------------------------------------------

    suspicious_speed = df["speed"] > 40

    sentinel_speed = np.isclose(
        df["speed"],
        102.3,
        atol=0.001,
    )

    print("\n========== SPEED ==========")
    print("Maximum speed:", df["speed"].max())
    print("Speed > 40 knots:", suspicious_speed.sum())
    print("Speed approximately 102.3:", sentinel_speed.sum())

    print("\nSpeed distribution:")
    print(df["speed"].describe())

    if sentinel_speed.sum() > 0:

        print("\nExample 102.3 speed records:")

        print(
            df.loc[
                sentinel_speed,
                [
                    "mmsi",
                    "segment_id",
                    "timestamp",
                    "speed",
                ],
            ].head(10)
        )

    # --------------------------------------------------
    # Stationary observations
    # --------------------------------------------------

    stationary_percentage = (
        (df["speed"] == 0).mean() * 100
    )

    print("\n========== STATIONARY ==========")
    print(
        f"Stationary observations: "
        f"{stationary_percentage:.2f}%"
    )

    # --------------------------------------------------
    # Sort trajectories
    # --------------------------------------------------

    df = df.sort_values(
        ["mmsi", "segment_id", "timestamp"]
    ).reset_index(drop=True)

    # --------------------------------------------------
    # Time gaps
    # --------------------------------------------------

    df["time_diff"] = (
        df.groupby(
            ["mmsi", "segment_id"]
        )["timestamp"].diff()
    )

    gaps = df["time_diff"].dropna()

    print("\n========== TIME GAPS ==========")
    print("Total gaps:", len(gaps))

    print("\nGap distribution:")
    print(gaps.describe())

    for minutes in [10, 20, 30]:

        count = (
            gaps == pd.Timedelta(minutes=minutes)
        ).sum()

        print(
            f"Exactly {minutes} minutes: {count}"
        )

    unexpected_gaps = gaps[
        gaps != pd.Timedelta(minutes=10)
    ]

    print(
        "\nGaps not exactly 10 minutes:",
        len(unexpected_gaps)
    )

    # --------------------------------------------------
    # Previous coordinates
    # --------------------------------------------------

    df["prev_lat"] = (
        df.groupby(
            ["mmsi", "segment_id"]
        )["lat"].shift(1)
    )

    df["prev_lon"] = (
        df.groupby(
            ["mmsi", "segment_id"]
        )["lon"].shift(1)
    )

    # --------------------------------------------------
    # Haversine movement
    # --------------------------------------------------

    valid_movement = df["prev_lat"].notna()

    df["distance_km"] = np.nan

    df.loc[
        valid_movement,
        "distance_km"
    ] = haversine_distance(
        df.loc[valid_movement, "prev_lat"],
        df.loc[valid_movement, "prev_lon"],
        df.loc[valid_movement, "lat"],
        df.loc[valid_movement, "lon"],
    )

    movement = df["distance_km"].dropna()

    print("\n========== MOVEMENT ==========")

    print(
        "Mean distance / timestep:",
        movement.mean(),
        "km",
    )

    print(
        "Median distance / timestep:",
        movement.median(),
        "km",
    )

    print(
        "95th percentile:",
        movement.quantile(0.95),
        "km",
    )

    print(
        "99th percentile:",
        movement.quantile(0.99),
        "km",
    )

    print(
        "Maximum:",
        movement.max(),
        "km",
    )

    # --------------------------------------------------
    # Extreme movement
    # --------------------------------------------------

    extreme_movement = (
        df["distance_km"] > 10
    )

    print(
        "\nPoints moving >10 km "
        "in one timestep:",
        extreme_movement.sum(),
    )

    if extreme_movement.sum() > 0:

        print(
            "\nExample extreme movements:"
        )

        print(
            df.loc[
                extreme_movement,
                [
                    "mmsi",
                    "segment_id",
                    "timestamp",
                    "prev_lat",
                    "prev_lon",
                    "lat",
                    "lon",
                    "speed",
                    "distance_km",
                ],
            ].head(10)
        )