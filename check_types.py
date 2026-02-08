import polars as pl

try:
    df = pl.read_csv("6_months_sells.csv")
    print("Sales dtype:", df['Sales'].dtype)
    print("AdCost dtype:", df['AdCost'].dtype)
    
    print("Sales head:", df['Sales'].head().to_list())
    print("AdCost head:", df['AdCost'].head().to_list())
except Exception as e:
    print(e)
