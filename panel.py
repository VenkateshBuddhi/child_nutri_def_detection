# import pyreadstat
# import pandas as pd

# pd.set_option("display.max_columns", None)  
# pd.set_option("display.width", 1000)        


# df, meta = pyreadstat.read_sav("IAKR7EFL.SAV", apply_value_formats=False)

# print("Shape:", df.shape)
# print("\nFirst 5 rows (all columns):")
# print(df.head())

import pyreadstat

# Read .sav
df, meta = pyreadstat.read_sav("IAKR7EFL.SAV", apply_value_formats=False)

# Convert to CSV
df.to_csv("IAKR7EFL.csv", index=False, encoding="utf-8")

print("Done: IAKR7EFL.csv created")