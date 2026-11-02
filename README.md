# eetool

Download Google Earth Engine data **onto your own pixel grid** — exact extent,
CRS and pixel size taken from a reference raster — fast (parallel
`computePixels` tiles), robust (retries, streaming to disk) and compact
(server-side integer encoding).

| Function | Dataset | Output |
|---|---|---|
| `ee_download_landsat_s2_stm` | Landsat 4/5/7/8/9 C2 L2 and/or Sentinel-2 L2A (A/B/C/…): cloud-masked per-pixel **percentiles** (spectral-temporal metrics), optional indices | int16, value × 10000, NoData −9999 |
| `ee_download_annual_embedding` | Google Satellite Embedding V1 annual (64 bands) | uint16, [−1, 1] → 0…65534, NoData 65535 |
| `ee_download_s1_composite` | Sentinel-1 GRD IW backscatter, mean/median/min/max per *N*-day window | uint16, dB range → 0…65534, NoData 65535 |

## Install

```bash
pip install git+https://github.com/vudongpham/eetool.git
```

For development:

```bash
pip install -e ".[dev]" && pytest
```

## Quick start

```python
import eetool
from eetool import (ee_download_landsat_s2_stm, ee_download_annual_embedding,
                    ee_download_s1_composite)

eetool.initialize(project="my-gcp-project")   # ee.Initialize + high-volume endpoint

ref = "reference.tif"

stm = ee_download_landsat_s2_stm(ref, year=2024, percentiles=[0.1, 0.5, 0.9],
                                 sensors=["L8", "L9"], indices=["ndvi"])
stm.to_raster("stm_2024.tif")

ee_download_annual_embedding(ref, year=2024, path_out="embedding_2024.tif")

s1 = ee_download_s1_composite(ref, year=2024, polarizations=["VV", "VH"],
                              window_days=10, orbit="ASCENDING")
s1_db = s1.decode()          # float32 dB, NaN where no data
```

See [`examples/quickstart.py`](examples/quickstart.py).

## One interface for every function

Every `ee_download_*` function takes the product's own parameters, and then
the same download options:

| Option | Meaning |
|---|---|
| `raster_ref_image` | Path to a reference raster (upper-left corner, extent and CRS define the grid) **or** an `eetool.Grid` |
| `res` | Pixel size in CRS units. Default: the product's (30 m STM, 10 m embedding / S1) |
| `path_out` | `None` → return a `RasterResult` in memory; a path → stream tiles into that GeoTIFF and return the path |
| `tile_size` | Pixels per request side (default 256; capped automatically under the 48 MB request limit) |
| `n_workers` | Parallel requests (use the high-volume endpoint for > 4) |
| `max_retries` | Retries per tile with exponential back-off (non-transient errors fail fast) |
| `verbose` | Scene counts, grid summary, progress/ETA |

Temporal products accept either `year=2024` or `start_date=..., end_date=...`
(ISO strings, **end exclusive**).

### `RasterResult`

```python
res.data          # np.ndarray (bands, H, W), encoded dtype
res.band_names    # ['p10_blue', ...] / ['A00', ...] / ['VV_2024-01-01', ...]
res.decode()      # float32 physical values, NaN = NoData
res.band("p50_nir", decode=True)
res.select(["p50_red", "p50_nir"])
res.to_raster("out.tif")   # band descriptions + decoding tags included
np.asarray(res)   # works like the array
```

GeoTIFFs carry `eetool_factor` / `eetool_offset` tags:
`value = stored / eetool_factor + eetool_offset`.

### Custom grids

```python
grid = eetool.Grid.from_bounds((500000, 4990000, 520000, 5000000), "EPSG:32633", res=20)
ee_download_s1_composite(grid, year=2024, path_out="s1.tif")
```

## Product notes

**`ee_download_landsat_s2_stm`** — `sensors` takes any mix of `L4 L5 L7 L8 L9`
and `S2A S2B S2C …` (`S2` = all); all scenes are pooled before percentiles.
Landsat uses QA_PIXEL (fill, dilated cloud, cirrus, cloud, shadow, optional
snow) + QA_RADSAT; Sentinel-2 uses Cloud Score+ (`s2_cs_threshold`, default
0.6) + SCL. Extras: `bands=` subset, `indices=` (`ndvi nbr ndmi ndwi mndwi
ndbi`), `months=` (seasonal composites, e.g. `[6, 7, 8]`), `max_scene_cloud=`,
`s2_nir="B8A"|"B8"`, `resample=`. Bands are percentile-major:
`p10_blue, p10_green, …, p50_blue, …`.

**`ee_download_annual_embedding`** — `year`, optional `bands=` subset of
`A00…A63`. 128 bytes/pixel for all 64 bands — prefer `path_out` for large areas.
`eetool.dequantize_embedding(arr)` decodes raw arrays.

**`ee_download_s1_composite`** — `polarizations` (`VV VH HH HV`), `orbit`
(`ASCENDING | DESCENDING | BOTH`), `window_days` (`None` = one composite),
`reducer` (`mean median min max`), `linear=True` to average in linear power,
`relative_orbits=`, `platforms=["A", "C"]`, `mask_border_noise` (masks
< −30 dB edge artefacts), `db_range` for storage (default −40…10 dB,
≈0.0008 dB precision). Bands are polarisation-major: `VV_2024-01-01, …, VH_2024-01-01, …`.

## Adding a new dataset

All tiling, retries, encoding and writing is shared; a product only describes
its image. Create `src/eetool/products/my_product.py`:

```python
from eetool.core import Encoding, Product, register, resolve_period, run


@register
class MonthlyNDVI(Product):
    name = "monthly_ndvi"          # usable via eetool.download("monthly_ndvi", ...)
    default_res = 10
    encoding = Encoding.scaled(10000, "int16", -9999)

    def __init__(self, year):
        self.period = resolve_period(year=year)

    @property
    def band_names(self):
        return [f"ndvi_{m:02d}" for m in range(1, 13)]

    def build_image(self, region):
        import ee
        ...  # ee.Image with len(band_names) bands, physical values, masked = NoData
        return image

    # optional: prepare(self, grid, verbose) for validation / scene counts,
    #           metadata(self) for GeoTIFF tags


def ee_download_monthly_ndvi(raster_ref_image, year, *, res=None, path_out=None,
                             tile_size=None, n_workers=4, max_retries=5, verbose=True):
    return run(MonthlyNDVI(year), raster_ref_image, res=res, path_out=path_out,
               tile_size=tile_size, n_workers=n_workers, max_retries=max_retries,
               verbose=verbose)
```

Then export it from `src/eetool/products/__init__.py` and `src/eetool/__init__.py`.

## Layout

```
src/eetool/
  core/
    grid.py       Grid + tiling + computePixels grid requests
    encoding.py   float <-> integer encodings (client and Earth Engine side)
    download.py   parallel tiled downloader, retries, streaming, download()
    product.py    Product base class + registry
    result.py     RasterResult
    io.py         GeoTIFF writing
    dates.py      periods and time windows
  products/
    landsat_s2.py   ee_download_landsat_s2_stm
    embedding.py    ee_download_annual_embedding
    sentinel1.py    ee_download_s1_composite
```
