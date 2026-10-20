import pandas as pd, numpy as np
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 20)

f = r"C:\Users\aselk\OneDrive\Desktop\De-anomalized data\Data-De-anomalyzation\ANOM_DS02_SiteLocations.csv"
df = pd.read_csv(f)
print(df.head(40).to_string())

key = df.LAT.astype(str) + "|" + df.LON.astype(str)
run = (key != key.shift()).cumsum()
df['run'] = run
print("\n--- rows 60..100 ---")
print(df.iloc[60:100].to_string())

# how many rows does each site have, and in how many blocks
b = (df.ANONYMIZED_SITECODE != df.ANONYMIZED_SITECODE.shift()).cumsum()
blocks = df.groupby('ANONYMIZED_SITECODE').apply(lambda g: pd.Series({
    'rows': len(g), 'blocks': (g.ANONYMIZED_SITECODE.index.to_series().diff() != 1).sum() + 1,
    'uniquecoords': g.LAT.astype(str).add(g.LON.astype(str)).nunique()}), include_groups=False)
print("\nrows per site:\n", blocks.rows.describe())
print("\nblocks per site:\n", blocks.blocks.describe())
print("\nunique coords per site:\n", blocks.uniquecoords.describe())

# within-site cell codes unique?
print("\nrows total", len(df), "unique cellcodes", df.ANOM_CELLCODE.nunique())
