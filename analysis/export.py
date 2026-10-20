"""Export the de-anonymised DS02 (row-level + site-level + map)."""
import json, os
import numpy as np, pandas as pd
from PIL import Image, ImageDraw

HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
T=json.load(open(os.path.join(HERE,"transform_tech.json")))
B=np.array(T["B"]); c=np.array(T["c"])

def load_rings(p):
    gj=json.load(open(p,encoding="utf-8")); out=[]
    for f in gj["features"]:
        g=f["geometry"]; polys=g["coordinates"] if g["type"]=="MultiPolygon" else [g["coordinates"]]
        for q in polys: out.append(np.array(q[0],float))
    return out
RING=max(load_rings(os.path.join(HERE,"qatar_land.geojson")),key=len)
res=0.0005; LON0,LON1,LAT0,LAT1=50.55,51.85,24.35,26.35
W=int(round((LON1-LON0)/res))+1; H=int(round((LAT1-LAT0)/res))+1
im=Image.new("L",(W,H),0); ImageDraw.Draw(im).polygon([((x-LON0)/res,(LAT1-y)/res) for x,y in RING],fill=255)
LAND=np.array(im)>0
def off_km(lon,lat):
    ix=np.clip(((lon-LON0)/res).round().astype(int),0,W-1)
    iy=np.clip(((LAT1-lat)/res).round().astype(int),0,H-1)
    bad=~LAND[iy,ix]
    if not bad.any(): return np.zeros(len(lon))
    P=np.c_[lon[bad],lat[bad]]
    d=np.min(np.hypot(P[:,0][:,None]-RING[None,:,0],P[:,1][:,None]-RING[None,:,1]),axis=1)*111.0
    out=np.zeros(len(lon)); out[bad]=d
    return out

df=pd.read_csv(os.path.join(ROOT,"ANOM_DS02_SiteLocations.csv"))
n=len(df)
df["LAT_SHIFTED"]=np.concatenate([df.LAT.values[1:],[np.nan]])
df["LON_SHIFTED"]=np.concatenate([df.LON.values[1:],[np.nan]])
shifted=df.dropna(subset=["LAT_SHIFTED"])
site=shifted.groupby("ANONYMIZED_SITECODE").agg(LON_A=("LON_SHIFTED","median"),
                                                LAT_A=("LAT_SHIFTED","median"),
                                                NCOORD=("LON_SHIFTED","nunique"))
print("sites %d | sites with >1 shifted coordinate: %d" % (len(site),(site.NCOORD>1).sum()))
miss=set(df.ANONYMIZED_SITECODE.dropna())-set(site.index)
print("sites with no recoverable coordinate:", len(miss), sorted(miss)[:5])

A=site[["LON_A","LAT_A"]].values
m=A@B.T+c
site["LON_REAL"]=m[:,0]; site["LAT_REAL"]=m[:,1]
site["OFF_LAND_KM"]=off_km(m[:,0],m[:,1])

# site attributes from DS02 (rows = cells)
code_sets=(df.dropna(subset=["LAC_TAC"]).groupby("ANONYMIZED_SITECODE").LAC_TAC
             .apply(lambda s: ";".join(str(int(v)) for v in sorted(set(s)))))
attrs=df.groupby("ANONYMIZED_SITECODE").agg(
    N_CELLS=("ANOM_CELLCODE","size"), CELLS_2G=("CELLS2G","max"), CELLS_4G=("CELLS4G","max"),
    CELLS_5G=("CELLS5G","max"), SITE_TYPE=("SITETYPE","first"),
    N_AZIMUTHS=("AZIMUTHSFORPLOT","nunique"), N_2GCODE=("LAC_TAC","count"))
anchored=code_sets.apply(lambda s: any(len(v)<=4 and v not in ("10101","14101","15101")
                                       for v in s.split(";") if v) )
site=site.join(attrs).join(code_sets.rename("LAC_TAC_CODES"))
site["GEO_METHOD"]=np.where(anchored,"anchored (OpenCelliD per-LAC)","unanchored (transform only)")
site=site.reset_index().rename(columns={"ANONYMIZED_SITECODE":"ANONYMIZED_SITECODE"})
site=site[["ANONYMIZED_SITECODE","LAT_REAL","LON_REAL","LON_A","LAT_A","OFF_LAND_KM","GEO_METHOD",
           "LAC_TAC_CODES","N_CELLS","CELLS_2G","CELLS_4G","CELLS_5G","SITE_TYPE","N_AZIMUTHS"]]
site.to_csv(os.path.join(HERE,"..","DS02_sites_deanonymised.csv"),index=False)
print("wrote DS02_sites_deanonymised.csv  (%d sites)" % len(site))

# row-level
row=df.drop(columns=["LAT_SHIFTED","LON_SHIFTED"]).merge(
    site[["ANONYMIZED_SITECODE","LAT_REAL","LON_REAL","GEO_METHOD"]],
    on="ANONYMIZED_SITECODE", how="left")
row=row.rename(columns={"LAT":"LAT_WARPED_FILE","LON":"LON_WARPED_FILE"})
row=row[["ANONYMIZED_SITECODE","ANOM_CELLCODE","LAT_REAL","LON_REAL","GEO_METHOD",
         "LAT_WARPED_FILE","LON_WARPED_FILE","AZIMUTHSFORPLOT","LAC_TAC","SITETYPE","CELLS2G","CELLS4G","CELLS5G"]]
row.to_csv(os.path.join(HERE,"..","DS02_cells_deanonymised.csv"),index=False)
print("wrote DS02_cells_deanonymised.csv  (%d cell rows)" % len(row))

# map
S=1500; LO0,LO1,LA0,LA1=50.55,51.80,24.40,26.30
def px(lon,lat): return ((lon-LO0)/(LO1-LO0)*S, S-(lat-LA0)/(LA1-LA0)*S)
img=Image.new("RGB",(S,S),"white"); dr=ImageDraw.Draw(img)
for g in np.arange(LO0,LO1,0.1):
    x,_=px(g,LA0); dr.line([(x,0),(x,S)],fill=(238,238,238))
for g in np.arange(LA0,LA1,0.1):
    _,y=px(LO0,g); dr.line([(0,y),(S,y)],fill=(238,238,238))
dr.polygon([px(x,y) for x,y in RING], fill=(214,232,246), outline=(40,80,170))
CITY={"Doha":(51.533,25.286),"AlWakrah":(51.60,25.17),"AlKhor":(51.51,25.68),"Dukhan":(50.78,25.42),
      "Mesaieed":(51.55,24.99),"AlRuwais":(51.21,26.13),"UmmBab":(50.80,25.20),"AbuSamra":(50.83,24.75),
      "Sealine":(51.55,24.85),"Shahaniya":(51.20,25.37),"Lusail":(51.49,25.43),"RasLaffan":(51.53,25.90),
      "Zubarah":(51.03,25.98),"Fuwayrit":(51.37,26.02),"Karaana":(51.05,25.05)}
for lon,lat in site[["LON_REAL","LAT_REAL"]].values:
    x,y=px(lon,lat)
    if 0<=x<S and 0<=y<S: dr.ellipse([x-2.6,y-2.6,x+2.6,y+2.6],fill=(200,0,0))
for k,(lo,la) in CITY.items():
    x,y=px(lo,la)
    dr.line([(x-6,y),(x+6,y)],fill=(0,110,0)); dr.line([(x,y-6),(x,y+6)],fill=(0,110,0))
    dr.text((x+7,y-7),k,fill=(0,90,0))
img.save(os.path.join(HERE,"final_map.png"))
print("wrote final_map.png")
print("\nsummary: %.1f%% of sites on land; %d sites >2 km offshore; median |off| among offshore %.1f km" %
      (100*(site.OFF_LAND_KM==0).mean(), int((site.OFF_LAND_KM>2).sum()),
       site.OFF_LAND_KM[site.OFF_LAND_KM>2].median() if (site.OFF_LAND_KM>2).any() else 0))
print("lon %.4f..%.4f  lat %.4f..%.4f" % (site.LON_REAL.min(),site.LON_REAL.max(),
                                          site.LAT_REAL.min(),site.LAT_REAL.max()))
