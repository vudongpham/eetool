# eetool

Google Earth Engine data as numpy arrays on the pixel grid of your own
raster or vector file. Every function returns a 4-D array
**`(time, height, width, channel)`**.

| Function | time axis | channel axis |
|---|---|---|
| `ee_get_landsat_s2_te` | one step per `time_step` days | bands |
| `ee_get_landsat_s2_stm` | one step per percentile (`p25`, `p50`, ...) | bands |
| `ee_get_s1_composite` | one step per `time_step` days | polarisations |
| `ee_get_annual_embedding` | 1 (the year) | `A00` ... `A63` |

## Sensors and bands

| `sensors=` | Satellite | Data from |
|---|---|---|
| `"L4"`, `"L5"` | Landsat 4 / 5 TM | 1982 / 1984 |
| `"L7"` | Landsat 7 ETM+ | 1999 |
| `"L8"`, `"L9"` | Landsat 8 / 9 OLI | 2013 / 2021 |
| `"S2A"`, `"S2B"`, `"S2C"` | Sentinel-2 A / B / C MSI (L2A) | 2017 / 2017 / 2025 |
| `"S2"` | all Sentinel-2 platforms | |

| Sensors selected | Default `bands` |
|---|---|
| Sentinel-2 only | 10: `blue green red re1 re2 re3 bnir nir swir1 swir2` (`bnir` = B8, `nir` = B8A) |
| any Landsat (alone or with S2) | 6: `blue green red nir swir1 swir2` |

Asking for `re1`, `re2`, `re3` or `bnir` while a Landsat sensor is selected
raises an error.

Sentinel-1 GRD (`ee_get_s1_composite`) and the Google Satellite Embedding
(`ee_get_annual_embedding`) have their own functions.

## Data sources

Each function reads these Earth Engine `ImageCollection`s:

| Function | `sensors=` / data | Earth Engine `ImageCollection` |
|---|---|---|
| `ee_get_landsat_s2_te`, `ee_get_landsat_s2_stm` | `"L4"` | [`LANDSAT/LT04/C02/T1_L2`](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LT04_C02_T1_L2) |
| | `"L5"` | [`LANDSAT/LT05/C02/T1_L2`](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LT05_C02_T1_L2) |
| | `"L7"` | [`LANDSAT/LE07/C02/T1_L2`](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LE07_C02_T1_L2) |
| | `"L8"` | [`LANDSAT/LC08/C02/T1_L2`](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC08_C02_T1_L2) |
| | `"L9"` | [`LANDSAT/LC09/C02/T1_L2`](https://developers.google.com/earth-engine/datasets/catalog/LANDSAT_LC09_C02_T1_L2) |
| | `"S2"`, `"S2A"`, ... | [`COPERNICUS/S2_SR_HARMONIZED`](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED) |
| | S2 cloud mask | [`GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED`](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_CLOUD_SCORE_PLUS_V1_S2_HARMONIZED) (band `cs_cdf`) |
| `ee_get_s1_composite` | Sentinel-1 GRD, IW mode | [`COPERNICUS/S1_GRD`](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD) |
| `ee_get_annual_embedding` | Satellite Embedding | [`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL) |

How each source is used:

- **Landsat** (Collection 2, Tier 1, Level-2 surface reflectance): bands
  `SR_B1 SR_B2 SR_B3 SR_B4 SR_B5 SR_B7` (TM / ETM+) or
  `SR_B2 SR_B3 SR_B4 SR_B5 SR_B6 SR_B7` (OLI) become
  `blue green red nir swir1 swir2`, scaled with `× 0.0000275 − 0.2`. Masked
  with `QA_PIXEL` (fill, dilated cloud, cirrus, cloud, shadow, and snow if
  `mask_snow=True`) and `QA_RADSAT` (saturated pixels). `max_scene_cloud`
  filters on `CLOUD_COVER`.
- **Sentinel-2** (L2A, harmonized: the 2022 processing-baseline offset is
  already removed): bands `B2 B3 B4 B5 B6 B7 B8 B8A B11 B12`, divided by 10000.
  Each scene is joined to its Cloud Score+ image and masked where
  `cs_cdf < s2_cs_threshold`, plus `SCL` classes 0 (no data), 1 (defective) and
  11 (snow, if `mask_snow=True`). `max_scene_cloud` filters on
  `CLOUDY_PIXEL_PERCENTAGE`; `S2A`/`S2B`/`S2C` filter on `SPACECRAFT_NAME`.
- **Sentinel-1**: GRD scenes in `instrumentMode = IW` with the requested
  polarisations (`transmitterReceiverPolarisation`), already calibrated and
  terrain-corrected to dB by Earth Engine. `orbit`, `relative_orbits` and
  `platforms` filter on `orbitProperties_pass`, `relativeOrbitNumber_start`
  and `platform_number`.
- **Satellite Embedding**: the annual images of `year` (UTM tiles) are mosaicked;
  bands `A00` ... `A63`.

## Install

```bash
pip install --upgrade --force-reinstall git+https://github.com/vudongpham/eetool.git
```

## Quick start

```python
import eetool
from eetool import ee_get_landsat_s2_te

eetool.initialize(project="my-gcp-project")

te = ee_get_landsat_s2_te("ref.gpkg", start_date="2025-01-01", end_date="2025-12-31",
                          time_step=30, sensors=["S2"], showProgress=True)
te.shape       # (13, H, W, 10)
te.times       # ['2025-01-01', '2025-01-31', ...]
te.channels    # ['blue', 'green', 'red', 're1', 're2', 're3', 'bnir', 'nir', 'swir1', 'swir2']
x = te.decode()   # float32 reflectance, NaN = no data
```

Large requests can take minutes. Use `showProgress=True` to see the tiles
arrive; by default nothing is printed.

## Global arguments (all functions)

| Argument | Default | Meaning |
|---|---|---|
| `reference` | required | Grid source: raster (`.tif`, ...), vector (`.gpkg`, `.shp`, ...), GeoDataFrame or `eetool.Grid` |
| `start_date`, `end_date` | required | `"20250101"` or `"2025-01-01"`, **both included** (`ee_get_annual_embedding` takes `year`) |
| `res` | `None` | Pixel size in CRS units. `None` = 30 for Landsat/S2, 10 for S1 and embedding |
| `path_out` | `None` | `None` = return a `RasterResult`. A path = write GeoTIFF(s) while downloading and return the path(s) |
| `layout` | `"stack"` | With `path_out`: `"stack"` = one file, time × channel bands; `"per_time"` = one file per time step (`te_2025-01-01.tif`, ...) |
| `showProgress` | `False` | Print grid size, scene counts and a progress bar |
| `tile_size` | `256` | Pixels per tile side per Earth Engine request |
| `n_workers` | `4` | Tiles requested in parallel |
| `max_retries` | `5` | Retries per tile |

**Reference grid:** a raster gives its extent and CRS. A vector gives the
extent (xmin, ymin, xmax, ymax) of all its features in its CRS; a lon/lat
vector is put on the UTM zone of its centre. The grid starts exactly at
(xmin, ymax). `.gpkg` and `.shp` need no extra library; other vector formats
need `pip install pyogrio`.

```python
ee_get_landsat_s2_te("ref.gpkg", "2025-01-01", "2025-12-31", time_step=30,
                     sensors="S2", path_out="te.tif", layout="per_time")
```

## Running several jobs in parallel

Earth Engine allows **40 concurrent requests per Cloud project** (the
`project=` in `eetool.initialize`), for the standard and the high-volume
endpoint. Each eetool job keeps at most `n_workers` tile requests in flight,
whatever the function, so:

**jobs in parallel × `n_workers` ≤ 40** (per project)

| Function | Requests in flight per job | Max parallel jobs, `n_workers=4` | Recommended |
|---|---|---|---|
| `ee_get_landsat_s2_te` | 4 | 10 | 8 |
| `ee_get_landsat_s2_stm` | 4 | 10 | 8 |
| `ee_get_s1_composite` | 4 | 10 | 8 |
| `ee_get_annual_embedding` | 4 | 10 | 8 |

| `n_workers` | 1 | 2 | 4 | 5 | 8 | 10 |
|---|---|---|---|---|---|---|
| max parallel jobs | 40 | 20 | 10 | 8 | 5 | 4 |

- Keep some headroom (e.g. 8 jobs × 4 workers = 32). `showProgress=True` adds a
  short query per job before the tiles start, and the Code Editor or other
  scripts on the same project use the same 40.
- Jobs can mix functions: only the total number of requests counts.
- Going over the limit returns "429 Too Many Requests". eetool retries a tile
  with increasing waits (1, 2, 4, 8, 16 s; `max_retries=5`), so it slows
  down. A tile fails only if it is still refused after the last retry.
- Jobs on different Cloud projects do not share the limit.
- Concurrency is not the only limit. Each project also has a monthly compute
  quota (EECU-hours: 150 for the noncommercial Community tier, 1,000 for
  Contributor). A project over its quota is put in a restricted mode with lower
  concurrency.
- The function changes how long each request takes, not how many run at once.
  Sentinel-2 with Cloud Score+ and many time steps (`ee_get_landsat_s2_te`) is
  the slowest per tile. If tiles time out or run out of memory, lower `tile_size`
  rather than `n_workers`.

## Function-specific arguments

### `ee_get_landsat_s2_te`: composites every `time_step` days

| Argument | Default | Meaning |
|---|---|---|
| `time_step` | `7` | Days per step; the last step stops at `end_date` |
| `sensors` | `("L4", "L5", "L7", "L8", "L9")` | See "Sensors and bands" |
| `bands` | `None` | See "Sensors and bands" |
| `reducer` | `"mean"` | `"mean"`, `"median"`, `"min"` or `"max"` of the clear observations in a step |
| `NoDataOut` | `-9999` | Value for no data (no clear observation, or no scene in a step). Integer in -32768..32767 |
| `max_scene_cloud` | `None` | Skip scenes with more cloud cover (%) than this |
| `s2_cs_threshold` | `0.6` | Sentinel-2 Cloud Score+ threshold (higher = stricter) |
| `s2_nir` | `"B8A"` | Landsat + S2 mixed only: S2 band used as `nir` (`"B8A"` or `"B8"`) |
| `mask_snow` | `True` | Also mask snow and ice |
| `resample` | `None` | `None` (nearest), `"bilinear"` or `"bicubic"` |

Output: int16, reflectance × 10000.

### `ee_get_landsat_s2_stm`: percentiles over the period

Same arguments as `ee_get_landsat_s2_te` except `time_step`, `reducer` and
`NoDataOut`, plus:

| Argument | Default | Meaning |
|---|---|---|
| `percentiles` | `(0.5,)` | Fractions 0..1, e.g. `[0.25, 0.5, 0.75]`; one time step each |

Output: int16, reflectance × 10000, NoData = -9999.

### `ee_get_s1_composite`: Sentinel-1 backscatter (dB) every `time_step` days

| Argument | Default | Meaning |
|---|---|---|
| `polarizations` | `("VV",)` | Any of `"VV"`, `"VH"`, `"HH"`, `"HV"` |
| `time_step` | `10` | Days per step, as in `ee_get_landsat_s2_te`; `None` = one composite |
| `orbit` | `"ASCENDING"` | `"ASCENDING"`, `"DESCENDING"` or `"BOTH"` |
| `reducer` | `"mean"` | `"mean"`, `"median"`, `"min"` or `"max"` |
| `linear` | `False` | Average in linear power instead of dB |
| `relative_orbits` | `None` | Only these relative orbits, e.g. `[117]` |
| `platforms` | `None` | Only these satellites, e.g. `["A", "C"]` |
| `mask_border_noise` | `True` | Mask values below -30 dB |
| `db_range` | `(-40, 10)` | dB range stored; values outside are clipped |

Output: **uint16**. `db_range` (default -40..10 dB) is mapped onto 0..65534,
so one step is 50 / 65534 ≈ 0.0008 dB. NoData = 65535.

Converting the stored values back to dB:

```python
import numpy as np
import eetool

s1 = eetool.ee_get_s1_composite("ref.gpkg", "2024-01-01", "2024-12-31",
                                 polarizations=["VV", "VH"], time_step=10)

# 1) from the result
db = s1.decode()                    # float32 dB, (time, H, W, pol), NaN = no data
vv = db[..., s1.channels.index("VV")]

# 2) any uint16 array, e.g. s1.data
db = eetool.dequantize_s1(s1.data)  # pass db_range=(lo, hi) if you changed it

# 3) the formula, without eetool
def s1_to_db(q, db_min=-40.0, db_max=10.0):
    q = np.asarray(q)
    db = q.astype("float32") * (db_max - db_min) / 65534 + db_min
    return np.where(q == 65535, np.nan, db)

# 4) from a GeoTIFF written with path_out=... (the file stores the scale itself)
import rasterio
with rasterio.open("s1.tif") as src:
    q = src.read()                  # (bands, H, W) uint16
    factor = float(src.tags()["eetool_factor"])
    offset = float(src.tags()["eetool_offset"])
    db = np.where(q == src.nodata, np.nan, q / factor + offset)
# or: eetool.read_raster("s1.tif").decode()
```

### `ee_get_annual_embedding`: Google Satellite Embedding

| Argument | Default | Meaning |
|---|---|---|
| `year` | required | Embedding year (replaces `start_date` / `end_date`) |
| `bands` | `None` | Subset of `"A00"` ... `"A63"`; `None` = all 64 |

Output: uint16 (`decode()` gives -1..1), NoData = 65535.

## Methods

The result (`RasterResult`):

| Method / attribute | Returns |
|---|---|
| `.data`, `np.asarray(res)` | `(time, H, W, channel)` stored integers |
| `.times`, `.channels` | axis labels |
| `.decode()` | float32 values, NaN = no data |
| `.valid()` | bool array, True where there is data |
| `.nodata`, `.transform`, `.crs` | to georeference or write it yourself |
| `.sel(time=..., channel=...)` | subset by label or index, stays 4-D |
| `.to_bands()` | `(time × channel, H, W)`, the layout rasterio writes |
| `.to_raster(path, layout="stack")` | write GeoTIFF(s), see `layout` above |
| `.to_xarray()` | `xarray.DataArray` (time, y, x, channel); needs xarray |

Module functions:

| Function | Does |
|---|---|
| `eetool.initialize(project=...)` | `ee.Initialize` on the high-volume endpoint |
| `eetool.read_raster(path_or_paths)` | read eetool GeoTIFF(s) back into a `RasterResult` |
| `eetool.Grid.from_vector(path, res, layer=, crs=, snap=)` | grid from a vector, with layer / CRS choice |
| `eetool.Grid.from_bounds((xmin, ymin, xmax, ymax), crs, res)` | grid from coordinates |
| `eetool.get("landsat_s2_te", reference, ...)` | call a product by name |
| `eetool.list_products()` | available product names |
