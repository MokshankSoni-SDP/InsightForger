"""
Debug script to check column types in the dataset
"""
import polars as pl

df = pl.read_csv("6_months_sells.csv")

print("Column Types:")
print("=" * 80)
for col in df.columns:
    print(f"{col:30} {df[col].dtype}")

print("\n" + "=" * 80)
print("Problematic columns (list types):")
for col in df.columns:
    if "list" in str(df[col].dtype).lower():
        print(f"  - {col}: {df[col].dtype}")
