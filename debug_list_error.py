"""
Check what's causing the list type error
"""
import polars as pl
import re

df = pl.read_csv("6_months_sells.csv")

print("Checking 'Our_Position' column...")
print(f"Type: {df['Our_Position'].dtype}")
print(f"Sample values: {df['Our_Position'].head(10).to_list()}")

print("\nChecking 'Category' column...")
print(f"Type: {df['Category'].dtype}")
print(f"Sample values: {df['Category'].unique().head(10).to_list()}")

# Test the exact filter from Phase 2
print("\nTesting filter expression...")
try:
    test_df = df.filter(
        (pl.col('Sales').is_not_null()) & 
        (pl.col('AdCost').is_not_null()) & 
        (pl.col('AdCost') > 0)
    )
    print(f"✓ Filter works! Filtered to {len(test_df)} rows")
except Exception as e:
    print(f"✗ Filter failed: {e}")

# Test group_by
print("\nTesting group_by...")
try:
    test_result = test_df.group_by(['Our_Position', 'Category']).agg([
        (pl.col('Sales') / pl.col('AdCost').replace(0, None)).alias('value'),
        pl.count().alias('sample_size')
    ])
    print(f"✓ Group_by works! Got {len(test_result)} groups")
    print(f"\nTop 5 results:")
    print(test_result.sort('value', descending=True).head(5))
except Exception as e:
    print(f"✗ Group_by failed: {e}")
    import traceback
    traceback.print_exc()
