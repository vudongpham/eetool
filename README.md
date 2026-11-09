# eetool

Get Google Earth Engine data as **numpy arrays** on the exact pixel grid of a
reference raster (same extent and CRS, pixel size of your choice).

Every function returns the same thing: a 4-D array shaped
**`(time, height, width, channel)`**, plus the grid (CRS, transform) needed to
georeference it.

| Function | time axis | channel axis |
|---|---|---|
| `ee_download_landsat_s2_te` | one step per `time_step` days | spectral bands |
| `ee_download_s1_composite` | one step per `window_days` days | polarisations |
| `ee_download_landsat_s2_stm` | one per percentile (`p10`, `p50`, ...) | spectral bands |
| `ee_download_annual_embedding` | 1 (the year) | `A00` ... `A63` |

## Install

```bash
pip install --upgrade --force-reinstall git+https://github.com/vudongpham/eetool.git
```

This also replaces an eetool that is already installed. To update only eetool
and leave its dependencies (numpy, rasterio, earthengine-api) as they are:

```bash
pip install --upgrade --force-reinstall --no-deps git+https://github.com/vudongpham/eetool.git
```

## Quick start

```python
import eetool
from eetool import (ee_download_landsat_s2_stm, ee_download_landsat_s2_te,
                    ee_download_annual_embedding, ee_download_s1_composite)

eetool.initialize(project="my-gcp-project")

ref = "reference.tif"   # must be in a projected CRS (metres)

# Landsat 8/9 weekly mean composites for 2015
te = ee_download_landsat_s2_te(ref, start_date="20150101", end_date="20151231",
                               time_step=7, sensors=["L8", "L9"])
te.data.shape        # (53, H, W, 6)   int16, NoData = -9999
te.times             # ['2015-01-01', '2015-01-08', ...]
te.channels          # ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
x = te.decode()      # float32 reflectance, NaN where there is no clear observation

# Landsat 8/9 percentiles for 2024
stm = ee_download_landsat_s2_stm(ref, "20240101", "20241231",
                                 percentiles=[0.1, 0.5, 0.9], sensors=["L8", "L9"])
stm.data.shape       # (3, H, W, 6)    one time step per percentile
stm.times            # ['p10', 'p50', 'p90']

# Satellite Embedding for 2024
emb = ee_download_annual_embedding(ref, year=2024)
emb.data.shape       # (1, H, W, 64)

# Sentinel-1 VV + VH, 10-day means
s1 = ee_download_s1_composite(ref, "20240101", "20241231",
                              polarizations=["VV", "VH"], window_days=10)
s1.data.shape        # (37, H, W, 2)
```

## The result: `RasterResult`

```python
res.data          # numpy array (time, H, W, channel), stored integers
np.asarray(res)   # the same array
res.times         # labels of the time axis (first day of each step)
res.channels      # labels of the channel axis
res.decode()      # float32 real values, NaN where there is no data
res.valid()       # bool array, True where there is data
res.nodata        # NoData value of res.data
res.transform     # affine transform (upper-left corner), for writing it yourself
res.crs           # CRS
res.sel(time="2015-01-08", channel=["red", "nir"])   # subset, stays 4-D
res.to_bands()    # (time * channel, H, W), the layout rasterio writes
res.to_xarray()   # xarray.DataArray (time, y, x, channel), needs xarray
```

Writing it yourself with rasterio:

```python
import rasterio
bands = res.to_bands()
with rasterio.open("out.tif", "w", driver="GTiff", width=bands.shape[2], height=bands.shape[1],
                   count=bands.shape[0], dtype=bands.dtype, crs=res.crs,
                   transform=res.transform, nodata=res.nodata) as dst:
    dst.write(bands)
```

## Dates

Landsat/Sentinel-2 and Sentinel-1 functions take `start_date` and `end_date`,
written `"20240101"` or `"2024-01-01"`. **Both dates are included**, so
`end_date="20241231"` includes 31 Dec 2024. Time steps start at `start_date`;
the last one stops at `end_date` and can be shorter. The embedding takes a `year`.

## Saving to GeoTIFF

A GeoTIFF band is 2-D, so a 4-D array is saved in one of two layouts:

| `layout` | Files | Bands per file |
|---|---|---|
| `"stack"` (default) | `out.tif` | time × channel, time-major: `blue_2015-01-01, green_2015-01-01, ..., blue_2015-01-08, ...` |
| `"per_time"` | `out_2015-01-01.tif`, `out_2015-01-08.tif`, ... | channel |

With one time step (embedding, or STM with one percentile), the band names are just the channel names.
For STM the "time" label is the percentile: `blue_p10, green_p10, ..., blue_p50, ...`.
Both layouts store the time and channel labels in the file tags, so
`eetool.read_raster` gives back the same 4-D array:

```python
# from memory
res.to_raster("te.tif")                       # -> "te.tif"
res.to_raster("te.tif", layout="per_time")    # -> ["te_2015-01-01.tif", ...]

# or stream straight to disk while downloading, without holding the whole
# array in memory (use this for large areas or long time series)
ee_download_landsat_s2_te(ref, "20150101", "20151231", time_step=7,
                          sensors=["L8", "L9"], path_out="te.tif", layout="per_time")

back = eetool.read_raster("te.tif")                          # stacked file
back = eetool.read_raster(sorted(glob.glob("te_*.tif")))     # per-time files
back.data.shape    # (53, H, W, 6)
```

For a real 4-D file format, use `res.to_xarray()` and then `.to_zarr()` or `.to_netcdf()`.

### Options (same for all functions)

| Argument | Default | Meaning |
|---|---|---|
| `raster_ref_image` | required | Reference raster path, or an `eetool.Grid` |
| `res` | `None` | Pixel size in metres. `None` = 30 for Landsat/S2, 10 for embedding and S1 |
| `path_out` | `None` | `None` = return a `RasterResult`. A path = stream to GeoTIFF(s) and return the path(s) |
| `layout` | `"stack"` | File layout when `path_out` is given: `"stack"` or `"per_time"` |
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
    res=None, path_out=None, layout="stack", tile_size=None, n_workers=4, max_retries=5, verbose=True,
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

**Output:** `(n_percentiles, H, W, n_bands)`, int16, reflectance × 10000,
NoData = -9999. The time axis holds the percentiles in the order given
(`res.times == ['p10', 'p25', 'p50', 'p75']`), the channel axis the bands.

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
    res=None, path_out=None, layout="stack", tile_size=None, n_workers=4, max_retries=5, verbose=True,
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

**Output:** `(n_steps, H, W, n_bands)`, int16, reflectance × 10000,
NoData = -9999. `res.times` holds the first day of each step.

```python
res = ee_download_landsat_s2_te(ref, "20150101", "20151231", time_step=7,
                                sensors=["L8", "L9"], bands=["red", "nir"])
x = res.decode()                      # (53, H, W, 2) float32, NaN = no data
ndvi = (x[..., 1] - x[..., 0]) / (x[..., 1] + x[..., 0])   # (53, H, W)
```

## `ee_download_annual_embedding`

```python
ee_download_annual_embedding(
    raster_ref_image,
    year,
    bands=None,
    res=None, path_out=None, layout="stack", tile_size=None, n_workers=4, max_retries=5, verbose=True,
)
```

| Argument | Meaning |
|---|---|
| `year` | Embedding year |
| `bands` | Subset of `"A00"` ... `"A63"`. `None` = all 64 |

**Output:** `(1, H, W, n_bands)`, uint16, where values -1 to 1 map to 0 to 65534, NoData = 65535.
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
    res=None, path_out=None, layout="stack", tile_size=None, n_workers=4, max_retries=5, verbose=True,
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

**Output:** `(n_windows, H, W, n_polarizations)`, uint16, where `db_range`
maps to 0 to 65534, NoData = 65535. `res.times` holds the first day of each
window and `res.channels` the polarisations. `res.decode()` gives dB.

## Other tools

```python
# Use a bounding box instead of a reference raster
grid = eetool.Grid.from_bounds((500000, 4990000, 520000, 5000000), "EPSG:32633", res=20)
s1 = ee_download_s1_composite(grid, "20240101", "20241231")

# Call any product by name
eetool.list_products()   # ['annual_embedding', 'landsat_s2_stm', 'landsat_s2_te', 's1_composite']
emb = eetool.download("annual_embedding", ref, year=2023)
```

`eetool.initialize(project=...)` runs `ee.Initialize` with the high-volume
endpoint, which is recommended when downloading tiles in parallel.
