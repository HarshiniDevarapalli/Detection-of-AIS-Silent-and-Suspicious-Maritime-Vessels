import pandas as pd

file = "data/raw/gfw/trawlers.csv"

# Read only 1,000 rows for inspection
df = pd.read_csv(file, nrows=1000)

print("COLUMNS:")
print(df.columns.tolist())

print("\nFIRST 5 ROWS:")
print(df.head())

print("\nDATA TYPES:")
print(df.dtypes)

print("\nSHAPE OF SAMPLE:")
print(df.shape)