"""GeoTIFF writing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

from .encoding import Encoding
from .grid import Grid


def geotiff_profile(grid: Grid, count: int, encoding: Encoding, **overrides) -> dict:
    block = 256 if min(grid.width, grid.height) >= 256 else None
    profile = {
        "driver": "GTiff",
        "height": grid.height,
        "width": grid.width,
        "count": count,
        "dtype": encoding.dtype,
        "crs": grid.crs,
        "transform": grid.transform,
        "nodata": encoding.nodata,
        "compress": "deflate",
        "predictor": 3 if encoding.dtype == "float32" else 2,
        "interleave": "pixel",
        "BIGTIFF": "IF_SAFER",
    }
    if block:
        profile.update(tiled=True, blockxsize=block, blockysize=block)
    profile.update(overrides)
    return profile


def encoding_tags(encoding: Encoding) -> dict:
    """Tags that let anyone decode the file: x = q / factor + offset."""
    return {
        "eetool_factor": repr(float(encoding.factor)),
        "eetool_offset": repr(float(encoding.offset)),
        "eetool_decode": "value = stored / eetool_factor + eetool_offset",
    }


def open_for_writing(path, grid: Grid, band_names, encoding: Encoding,
                     tags: dict | None = None, **profile_overrides):
    """Open a GeoTIFF for writing with band descriptions and tags set."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dst = rasterio.open(path, "w", **geotiff_profile(grid, len(band_names), encoding,
                                                    **profile_overrides))
    for i, name in enumerate(band_names, start=1):
        dst.set_band_description(i, str(name))
    dst.update_tags(**encoding_tags(encoding), **{k: str(v) for k, v in (tags or {}).items()})
    return dst


def write_raster(path, data: np.ndarray, grid: Grid, band_names, encoding: Encoding,
                 tags: dict | None = None, **profile_overrides) -> str:
    """Write a (bands, height, width) array on ``grid``."""
    data = np.asarray(data)
    if data.ndim == 2:
        data = data[None]
    if data.shape != (len(band_names), grid.height, grid.width):
        raise ValueError(
            f"array shape {data.shape} does not match "
            f"({len(band_names)} bands, {grid.height}, {grid.width})"
        )
    with open_for_writing(path, grid, band_names, encoding, tags, **profile_overrides) as dst:
        dst.write(data.astype(encoding.dtype, copy=False))
    return str(path)
