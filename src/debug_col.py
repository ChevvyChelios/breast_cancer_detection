import pandas as pd

df = pd.read_csv("data/test_images/test_metadata.csv")

print("RAW columns:")
for col in df.columns:
    print(f"[{col}]  length={len(col)}")

print("\nAfter strip + lower:")
df.columns = df.columns.str.strip().str.lower()
for col in df.columns:
    print(f"[{col}]  length={len(col)}")
