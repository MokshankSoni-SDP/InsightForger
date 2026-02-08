"""
Diagnostic script for Polars list error
"""
import polars as pl
import traceback

try:
    print("Loading dataset...")
    df = pl.read_csv("6_months_sells.csv")
    print(f"Schema: {df.schema}")
    
    print("\nSanitizing...")
    sanitized = df.filter(
        (pl.col('Sales').is_not_null()) & 
        (pl.col('AdCost').is_not_null()) & 
        (pl.col('AdCost') > 0)
    )
    print(f"Sanitized rows: {len(sanitized)}")
    
    print("\nAttempting aggregation...")
    expr = (pl.col('Sales') / pl.col('AdCost').replace(0, None)).alias('value')
    
    # Try just the select first
    print("Testing select...")
    temp = sanitized.select([expr])
    print(f"Select result schema: {temp.schema}")
    print(f"Head: {temp.head(1)}")
    
    print("\nTesting Group By...")
    result_df = sanitized.group_by(['Our Position', 'SubCategory']).agg([
        expr,
        pl.count().alias('sample_size')
    ])
    print("Group By successful.")
    print(f"Result schema: {result_df.schema}")
    
    print("\nTesting Sort...")
    sorted_df = result_df.sort('value', descending=True)
    print("Sort successful.")
    
    print("\nTesting to_dict/extraction...")
    print(f"Top row: {sorted_df.head(1)}")
    # Validating extraction methods
    row = sorted_df.row(0, named=True)
    print(f"Row extracted: {row}")
    
except Exception:
    traceback.print_exc()
