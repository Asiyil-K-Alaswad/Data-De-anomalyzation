import pandas as pd
import numpy as np

f = r"C:\Users\aselk\OneDrive\Desktop\De-anomalized data\Data-De-anomalyzation\ANOM_DS02_SiteLocations.csv"
df = pd.read_csv(f)
print("shape", df.shape)
print(df.dtypes)
print(df.head())
print("\nunique sites", df.ANONYMIZED_SITECODE.nunique(), "unique cells", df.ANOM_CELLCODE.nunique())
print("\nLAT describe\n", df.LAT.describe())
print("\nLON describe\n", df.LON.describe())
print("\nlat span", df.LAT.max() - df.LAT.min(), "lon span", df.LON.max() - df.LON.min())
print("\nLAC_TAC value counts\n", df.LAC_TAC.value_counts().head(20))
print("\nSITETYPE\n", df.SITETYPE.value_counts())
print("\nCELLS2G/4G/5G\n", df[["CELLS2G", "CELLS4G", "CELLS5G"]].describe())
print("\nAZIMUTHSFORPLOT\n", df.AZIMUTHSFORPLOT.value_counts().head(20))
# unique coords
coords = df[["LAT", "LON"]].drop_duplicates()
print("\nunique coords", len(coords))
print(coords.sort_values(["LAT", "LON"]).to_string())
