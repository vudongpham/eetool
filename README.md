# eetool

Download Google Earth Engine data onto the exact pixel grid of a reference
raster (same extent and CRS, pixel size of your choice).

- `ee_download_landsat_s2_stm`: percentile composites of Landsat and/or Sentinel-2
- `ee_download_landsat_s2_te`: regular time series (e.g. weekly composites) of Landsat and/or Sentinel-2
- `ee_download_annual_embedding`: Google Satellite Embedding (64 bands)
- `ee_download_s1_composite`: Sentinel-1 backscatter composites over N-day windows

## Install

```bash
pip install git+https://github.com/vudongpham/eetool.git
```

## Quick start

```python
import eetool
from eetool import (ee_download_landsat_s2_stm, ee_download_landsat_s2_te,
                    ee_download_annual_embedding, ee_download_s1_composite)

eetool.initialize(project="my-gcp-project")

ref = "reference.tif"   # must be in a projected CRS (metres)

# Landsat 8/9 percentiles for 2024, saved to a GeoTIFF
ee_download_landsat_s2_stm(ref, start_date="20240101", end_date="20241231",
                           percentiles=[0.1, 0.5, 0.9], sensors=["L8", "L9"],
                           path_out="landsat_2024.tif")

# Landsat 8/9 weekly mean composites for 2015 (NoData where nothing is clear)
ee_download_landsat_s2_te(ref, start_date="20150101", end_date="20151231",
                          time_step=7, sensors=["L8", "L9"], path_out="landsat_te_2015.tif")

# Satellite Embedding for 2024
ee_download_annual_embedding(ref, year=2024, path_out="embedding_2024.tif")

# Sentinel-1 VV + VH, 10-day means
ee_download_s1_composite(ref, start_date="20240101", end_date="20241231",
                         polarizations=["VV", "VH"], window_days=10, path_out="s1_2024.tif")
```

## How every function works

- Landsat/Sentinel-2 and Sentinel-1 functions take a `start_date` / `end_date`
  pair, written `"20240101"` or `"2024-01-01"`. **Both dates are included**,
  so `end_date="20241231"` includes 31 Dec 2024. The embedding takes a `year`.
- `path_out="file.tif"` writes the result to a GeoTIFF and returns its path.
- Without `path_out`, you get the result in memory as a `RasterResult`:

```python
res = ee_download_s1_composite(ref, "20240101", "20241231")

res.data           # numpy array (bands, height, width), stored integers
res.band_names     # ['VV_2024-01-01', 'VV_2024-01-11', ...]
res.decode()       # float32 real values, NaN where there is no data
res.to_raster("s1_2024.tif")
```

### Download options (same for all functions)

| Argument | Default | Meaning |
|---|---|---|
| `raster_ref_image` | required | Reference raster path, or an `eetool.Grid` |
| `res` | `None` | Pixel size in metres. `None` = 30 for Landsat/S2, 10 for embedding and S1 |
| `path_out` | `None` | GeoTIFF to write. `None` = return a `RasterResult` |
| `tile_size` | `None` | Tile width/height in pixels per request (`None` = 256) |
| `n_workers` | `4` | Number of tiles downloaded in parallel |
| `max_retries` | `5` | Retries per tile when Earth Engine fails |
| `verbose` | `True` | Print scene counts and progress |

## `ee_download_landsat_s2_stm`

Cloud-masked percentile composite of surface reflectance. Scenes from all
selected sensors are pooled before the percentiles are computed.

```python
ee_download_landsat_s2_stm(
    raster_ref_image,
    start_date,
    end_date,
    percentiles=(0.5,),
    sensors=("L4", "L5", "L7", "L8", "L9"),
    bands=None,
    max_scene_cloud=None,
    s2_nir="B8A",
    s2_cs_threshold=0.6,
    mask_snow=True,
    resample=None,
    res=None, path_out=None, tile_size=None, n_workers=4, max_retries=5, verbose=True,
)
```

| Argument | Meaning |
|---|---|
| `percentiles` | Fractions between 0 and 1, e.g. `[0.1, 0.5, 0.9]` (0 = min, 1 = max) |
| `sensors` | Any of `"L4"`, `"L5"`, `"L7"`, `"L8"`, `"L9"`, `"S2A"`, `"S2B"`, `"S2C"`, ... `"S2"` = all Sentinel-2 |
| `bands` | Subset of `blue`, `green`, `red`, `nir`, `swir1`, `swir2`. `None` = all six |
| `max_scene_cloud` | Skip whole scenes with more cloud cover (%) than this |
| `s2_nir` | Sentinel-2 band used as nir: `"B8A"` (closer to Landsat) or `"B8"` |
| `s2_cs_threshold` | Sentinel-2 Cloud Score+ threshold, higher = stricter (0.5 to 0.65 is typical) |
| `mask_snow` | Also mask snow and ice |
| `resample` | `None` (nearest), `"bilinear"` or `"bicubic"`. Useful for Sentinel-2 at 30 m |

**Output:** int16, reflectance × 10000, NoData = -9999. Bands are named
`p10_blue, p10_green, ..., p50_blue, ...`.

## `ee_download_landsat_s2_te`

Temporal encoding: a regular time series of cloud-masked composites, one per
`time_step` days. Uses the same scenes, cloud masking and options as
`ee_download_landsat_s2_stm`, but instead of percentiles over the whole
period it combines the clear observations inside each time step.

```python
ee_download_landsat_s2_te(
    raster_ref_image,
    start_date,
    end_date,
    time_step=7,
    sensors=("L4", "L5", "L7", "L8", "L9"),
    bands=None,
    reducer="mean",
    max_scene_cloud=None,
    s2_nir="B8A",
    s2_cs_threshold=0.6,
    mask_snow=True,
    resample=None,
    res=None, path_out=None, tile_size=None, n_workers=4, max_retries=5, verbose=True,
)
```

| Argument | Meaning |
|---|---|
| `time_step` | Length of each step in days. Steps start at `start_date`; the last one stops at `end_date` and can be shorter |
| `reducer` | How clear observations inside a step are combined: `"mean"`, `"median"`, `"min"` or `"max"` |
| others | Same as `ee_download_landsat_s2_stm` |

A pixel gets NoData (-9999) in a step when it has no clear observation there:
it was cloudy, shadowed, snow-covered or saturated in every scene, or there
was no scene at all in that step (then the whole step is NoData).

**Output:** int16, reflectance × 10000, NoData = -9999. Bands are ordered by
time step, then band: `blue_2015-01-01, green_2015-01-01, ..., blue_2015-01-08, ...`
(the date is the first day of the step). To get a (time, band, H, W) cube:

```python
res = ee_download_landsat_s2_te(ref, "20150101", "20151231", time_step=7,
                                sensors=["L8", "L9"], bands=["red", "nir"])
cube = res.decode().reshape(-1, 2, *res.data.shape[1:])   # (53, 2, H, W), NaN = no data
```

## `ee_download_annual_embedding`

```python
ee_download_annual_embedding(
    raster_ref_image,
    year,
    bands=None,
    res=None, path_out=None, tile_size=None, n_workers=4, max_retries=5, verbose=True,
)
```

| Argument | Meaning |
|---|---|
| `year` | Embedding year |
| `bands` | Subset of `"A00"` ... `"A63"`. `None` = all 64 |

**Output:** uint16, where values -1 to 1 map to 0 to 65534, NoData = 65535.
`res.decode()` or `eetool.dequantize_embedding(array)` gives float values
back. All 64 bands take 128 bytes per pixel, so use `path_out` for large areas.

## `ee_download_s1_composite`

Sentinel-1 GRD (IW mode) backscatter in dB, one band per polarisation and
time window.

```python
ee_download_s1_composite(
    raster_ref_image,
    start_date,
    end_date,
    polarizations=("VV",),
    orbit="ASCENDING",
    window_days=10,
    reducer="mean",
    linear=False,
    relative_orbits=None,
    platforms=None,
    mask_border_noise=True,
    db_range=(-40.0, 10.0),
    res=None, path_out=None, tile_size=None, n_workers=4, max_retries=5, verbose=True,
)
```

| Argument | Meaning |
|---|---|
| `polarizations` | Any of `"VV"`, `"VH"`, `"HH"`, `"HV"` |
| `orbit` | `"ASCENDING"`, `"DESCENDING"` or `"BOTH"` |
| `window_days` | Length of each time window in days, starting at `start_date`. `None` = one composite for the whole period |
| `reducer` | `"mean"`, `"median"`, `"min"` or `"max"` |
| `linear` | `True` = average in linear power instead of dB (only affects `"mean"`) |
| `relative_orbits` | Only use these relative orbit numbers, e.g. `[117]` |
| `platforms` | Only use these satellites, e.g. `["A", "C"]` |
| `mask_border_noise` | Mask values below -30 dB (noise at swath edges) |
| `db_range` | dB range kept when storing values; values outside it are clipped |

**Output:** uint16, where `db_range` maps to 0 to 65534, NoData = 65535.
Bands are named `VV_2024-01-01, VV_2024-01-11, ..., VH_2024-01-01, ...`.
`res.decode()` gives dB.

## Other tools

```python
# Use a bounding box instead of a reference raster
grid = eetool.Grid.from_bounds((500000, 4990000, 520000, 5000000), "EPSG:32633", res=20)
ee_download_s1_composite(grid, "20240101", "20241231", path_out="s1.tif")

# Call any product by name
eetool.list_products()   # ['annual_embedding', 'landsat_s2_stm', 'landsat_s2_te', 's1_composite']
eetool.download("annual_embedding", ref, year=2023, path_out="emb_2023.tif")
```

`eetool.initialize(project=...)` runs `ee.Initialize` with the high-volume
endpoint, which is recommended when downloading tiles in parallel.
