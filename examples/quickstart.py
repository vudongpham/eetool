"""End-to-end example: four products as (time, H, W, channel) arrays on one grid."""

import eetool
from eetool import (ee_download_annual_embedding, ee_download_landsat_s2_stm,
                    ee_download_landsat_s2_te, ee_download_s1_composite)

eetool.initialize(project="my-gcp-project")  # high-volume endpoint by default

ref = "reference.tif"  # any raster in a projected CRS; defines extent + CRS

# 1) Landsat 8/9 weekly mean composites (temporal encoding); NoData = -9999
#    wherever a week has no clear observation
te = ee_download_landsat_s2_te(
    ref, start_date="20150101", end_date="20151231", time_step=7, sensors=["L8", "L9"],
)
print(te)                       # RasterResult(shape=(53, H, W, 6) (time, H, W, channel), ...)
print(te.times[:2], te.channels)
x = te.decode()                 # float32 reflectance, NaN = no data
red_nir = te.sel(channel=["red", "nir"]).data   # (53, H, W, 2)

# 2) Landsat 8/9 percentiles over three years -> (1, H, W, 18)
stm = ee_download_landsat_s2_stm(
    ref, start_date="20230101", end_date="20251231",
    percentiles=[0.1, 0.5, 0.75], sensors=["L8", "L9"],
)
print(stm.channels[:3])         # ['p10_blue', 'p10_green', 'p10_red']

# 3) Satellite Embedding 2024 at 10 m -> (1, H, W, 64)
#    (large: stream straight to disk instead of holding it in memory)
ee_download_annual_embedding(ref, year=2024, path_out="embedding_2024.tif")
emb = eetool.read_raster("embedding_2024.tif")

# 4) Sentinel-1 15-day mean VV + VH, ascending -> (25, H, W, 2)
s1 = ee_download_s1_composite(
    ref, "20240101", "20241231", polarizations=["VV", "VH"], window_days=15,
    orbit="ASCENDING",
)
vv_db = s1.decode()[..., 0]     # (25, H, W) float32 dB, NaN = no data

# Saving: one stacked GeoTIFF, or one file per time step
te.to_raster("landsat_te_2015.tif")                       # 53 * 6 bands
s1.to_raster("s1_2024.tif", layout="per_time")            # s1_2024_2024-01-01.tif, ...

# Any product by name, same options:
emb23 = eetool.download("annual_embedding", ref, year=2023, res=20)
