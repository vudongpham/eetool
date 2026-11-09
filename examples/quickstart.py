"""End-to-end example: four products on the same reference grid."""

import eetool
from eetool import (ee_download_annual_embedding, ee_download_landsat_s2_stm,
                    ee_download_landsat_s2_te, ee_download_s1_composite)

eetool.initialize(project="my-gcp-project")  # high-volume endpoint by default

ref = "reference.tif"  # any raster in a projected CRS; defines extent + CRS

# 1) Landsat 8/9 percentiles, 30 m, in memory -> GeoTIFF
stm = ee_download_landsat_s2_stm(
    ref, start_date="20230101", end_date="20251231",
    percentiles=[0.1, 0.5, 0.75], sensors=["L8", "L9"],
)
print(stm)                      # RasterResult(shape=(18, H, W), dtype=int16, ...)
print(stm.band_names[:3])       # ['p10_blue', 'p10_green', 'p10_red']
stm.to_raster("landsat_stm_2023_2025.tif")

# 2) Sentinel-2 A/B/C on the same 30 m grid, streamed straight to disk
ee_download_landsat_s2_stm(
    ref, "20240101", "20241231", percentiles=[0.25, 0.5, 0.75], sensors=["S2A", "S2B", "S2C"],
    resample="bilinear", path_out="s2_stm_2024.tif",
)

# 3) Landsat 8/9 weekly mean composites (temporal encoding); NoData = -9999
#    wherever a week has no clear observation
te = ee_download_landsat_s2_te(
    ref, start_date="20150101", end_date="20151231", time_step=7, sensors=["L8", "L9"],
)
cube = te.data.reshape(-1, 6, *te.data.shape[1:])   # (53 weeks, 6 bands, H, W)
te.to_raster("landsat_te_2015_7d.tif")

# 4) Satellite Embedding 2024 at 10 m (big: stream to disk)
ee_download_annual_embedding(ref, year=2024, path_out="embedding_2024.tif")

# 5) Sentinel-1 15-day mean VV + VH, ascending
s1 = ee_download_s1_composite(
    ref, "20240101", "20241231", polarizations=["VV", "VH"], window_days=15, orbit="ASCENDING",
)
vv_db = s1.decode()[: len(s1.band_names) // 2]   # float32 dB, NaN = no data
s1.to_raster("s1_2024_15d_asc.tif")

# Any product by name, same options:
eetool.download("annual_embedding", ref, year=2023, res=20, path_out="emb_2023_20m.tif")
