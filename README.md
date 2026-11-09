# eetool

Google Earth Engine data as numpy arrays on the pixel grid of your own
raster or vector file. Every function returns a 4-D array
**`(time, height, width, channel)`**.

| Function | time axis | channel axis |
|---|---|---|
| `ee_get_landsat_s2_te` | one step per `time_step` days | bands |
| `ee_get_landsat_s2_stm` | one step per percentile (`p25`, `p50`, ...) | bands |
| `ee_get_s1_composite` | one step per `window_days` days | polarisations |
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

### `ee_get_s1_composite`: Sentinel-1 backscatter (dB) every `window_days`

| Argument | Default | Meaning |
|---|---|---|
| `polarizations` | `("VV",)` | Any of `"VV"`, `"VH"`, `"HH"`, `"HV"` |
| `window_days` | `10` | Days per window; `None` = one composite |
| `orbit` | `"ASCENDING"` | `"ASCENDING"`, `"DESCENDING"` or `"BOTH"` |
| `reducer` | `"mean"` | `"mean"`, `"median"`, `"min"` or `"max"` |
| `linear` | `False` | Average in linear power instead of dB |
| `relative_orbits` | `None` | Only these relative orbits, e.g. `[117]` |
| `platforms` | `None` | Only these satellites, e.g. `["A", "C"]` |
| `mask_border_noise` | `True` | Mask values below -30 dB |
| `db_range` | `(-40, 10)` | dB range stored; values outside are clipped |

Output: uint16 (`decode()` gives dB), NoData = 65535.

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
