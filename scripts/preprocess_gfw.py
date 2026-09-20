import sys
import os
import pandas as pd

# --------------------------------------------------
# 1. Get dataset name
# --------------------------------------------------

if len(sys.argv) != 2:
    print("Usage:")
    print("python scripts/preprocess_gfw.py <dataset_name>")
    print("\nExample:")
    print("python scripts/preprocess_gfw.py drifting_longlines")
    sys.exit(1)

DATASET = sys.argv[1]

INPUT_FILE = f"data/raw/gfw/{DATASET}.csv"
OUTPUT_FILE = f"data/processed/{DATASET}_clean.csv"

# Check input exists
if not os.path.exists(INPUT_FILE):
    print(f"Error: File not found: {INPUT_FILE}")
    sys.exit(1)

# --------------------------------------------------
# 2. Load data
# --------------------------------------------------

print(f"Loading: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

print("Original shape:", df.shape)

# --------------------------------------------------
# 3. Convert timestamp
# --------------------------------------------------

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    unit="s",
    utc=True
)

# --------------------------------------------------
# 4. Convert vessel ID
# --------------------------------------------------

df["mmsi"] = (
    df["mmsi"]
    .astype("Int64")
    .astype(str)
)

# --------------------------------------------------
# 5. Sort
# --------------------------------------------------

df = df.sort_values(
    ["mmsi", "timestamp"]
)

# --------------------------------------------------
# 6. Remove invalid observations
# --------------------------------------------------

df = df.dropna(
    subset=[
        "mmsi",
        "timestamp",
        "lat",
        "lon",
        "speed",
        "course"
    ]
)

# --------------------------------------------------
# 7. Remove duplicates
# --------------------------------------------------

df = df.drop_duplicates(
    subset=[
        "mmsi",
        "timestamp",
        "lat",
        "lon"
    ]
)

# --------------------------------------------------
# 8. Save
# --------------------------------------------------

print("Cleaned shape:", df.shape)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print(f"Saved cleaned dataset to: {OUTPUT_FILE}")