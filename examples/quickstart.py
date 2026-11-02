"""End-to-end example: three products on the same reference grid."""

import eetool
from eetool import (ee_download_annual_embedding, ee_download_landsat_s2_stm,
                    ee_download_s1_composite)

eetool.initialize(project="my-gcp-project")  # high-volume endpoint by default

ref = "reference.tif"  # any raster in a projected CRS; defines extent + CRS

# 1) Landsat 8/9 percentiles + NDVI, 30 m, in memory -> GeoTIFF
stm = ee_download_landsat_s2_stm(
    ref, start_date="2023-01-01", end_date="2026-01-01",
    percentiles=[0.1, 0.5, 0.75], sensors=["L8", "L9"], indices=["ndvi"],
)
print(stm)                      # RasterResult(shape=(21, H, W), dtype=int16, ...)
print(stm.band_names[:3])       # ['p10_blue', 'p10_green', 'p10_red']
stm.to_raster("landsat_stm_2023_2025.tif")

# 2) Sentinel-2 A/B/C on the same 30 m grid, streamed straight to disk
ee_download_landsat_s2_stm(
    ref, year=2024, percentiles=[0.25, 0.5, 0.75], sensors=["S2A", "S2B", "S2C"],
    resample="bilinear", path_out="s2_stm_2024.tif",
)

# 3) Satellite Embedding 2024 at 10 m (big: stream to disk)
ee_download_annual_embedding(ref, year=2024, path_out="embedding_2024.tif")

# 4) Sentinel-1 15-day mean VV + VH, ascending
s1 = ee_download_s1_composite(
    ref, year=2024, polarizations=["VV", "VH"], window_days=15, orbit="ASCENDING",
)
vv_db = s1.decode()[: len(s1.band_names) // 2]   # float32 dB, NaN = no data
s1.to_raster("s1_2024_15d_asc.tif")

# Any product by name, same options:
eetool.download("annual_embedding", ref, year=2023, res=20, path_out="emb_2023_20m.tif")
