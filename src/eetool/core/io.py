"""
GeoTIFF writing and reading for 4-D (time, height, width, channel) arrays.

GeoTIFF bands are 2-D, so a 4-D array is stored in one of two layouts:

``"stack"`` (default)
    One file with ``time * channel`` bands in time-major order
    (all channels of time 0, then time 1, ...). Band descriptions are
    ``"{channel}_{time}"`` and the tags ``eetool_times`` / ``eetool_channels``
    record the two axes, so :func:`read_raster` rebuilds the 4-D array.
``"per_time"``
    One file per time step, ``<stem>_<time><suffix>``, each with ``channel``
    bands. Handy for long time series or tools that expect one date per file.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

from .encoding import Encoding
from .grid import Grid

LAYOUTS = ("stack", "per_time")


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


def _check_layout(layout):
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")


def _safe(label: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", str(label))


def output_paths(path, times, layout: str) -> list[Path]:
    """Files that ``layout`` produces for ``path``."""
    _check_layout(layout)
    path = Path(path)
    if layout == "stack":
        return [path]
    suffix = path.suffix or ".tif"
    return [path.with_name(f"{path.stem if path.suffix else path.name}_{_safe(t)}{suffix}")
            for t in times]


def _flat_names(times, channels):
    from .product import flat_band_names

    return flat_band_names(times, channels)


class RasterWriter:
    """
    Write a (time, H, W, channel) array block by block (``write(tile, block)``)
    into the GeoTIFF(s) of ``layout``. Files are written as ``*.part`` and
    renamed on success; on error they are removed. Use as a context manager.
    """

    def __init__(self, path, grid: Grid, times, channels, encoding: Encoding,
                 tags: dict | None = None, layout: str = "stack", **profile_overrides):
        self.paths = output_paths(path, times, layout)
        self.layout = layout
        self.encoding = encoding
        self.n_channels = len(channels)
        base_tags = {**encoding_tags(encoding),
                     **{k: str(v) for k, v in (tags or {}).items()},
                     "eetool_channels": json.dumps(list(channels)),
                     "eetool_layout": layout}
        self._tmp = [p.with_name(p.name + ".part") for p in self.paths]
        self._dst = []
        try:
            if layout == "stack":
                names = _flat_names(times, channels)
                self._open(self._tmp[0], grid, names,
                           {**base_tags, "eetool_times": json.dumps(list(times))},
                           profile_overrides)
            else:
                for tmp, t in zip(self._tmp, times):
                    self._open(tmp, grid, channels,
                               {**base_tags, "eetool_times": json.dumps([t])},
                               profile_overrides)
        except BaseException:
            self.abort()
            raise

    def _open(self, path, grid, band_names, tags, overrides):
        path.parent.mkdir(parents=True, exist_ok=True)
        dst = rasterio.open(path, "w", **geotiff_profile(grid, len(band_names),
                                                         self.encoding, **overrides))
        self._dst.append(dst)
        for i, name in enumerate(band_names, start=1):
            dst.set_band_description(i, str(name))
        dst.update_tags(**tags)

    def write(self, row: int, col: int, block: np.ndarray):
        """``block``: (time, h, w, channel) placed at pixel (row, col)."""
        t, h, w, c = block.shape
        block = block.astype(self.encoding.dtype, copy=False)
        window = Window(col, row, w, h)
        if self.layout == "stack":
            self._dst[0].write(block.transpose(0, 3, 1, 2).reshape(t * c, h, w), window=window)
        else:
            for i, dst in enumerate(self._dst):
                dst.write(block[i].transpose(2, 0, 1), window=window)

    def close(self) -> list[str]:
        for dst in self._dst:
            dst.close()
        for tmp, final in zip(self._tmp, self.paths):
            os.replace(tmp, final)
        return [str(p) for p in self.paths]

    def abort(self):
        for dst in self._dst:
            try:
                dst.close()
            except Exception:
                pass
        for tmp in self._tmp:
            tmp.unlink(missing_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is not None:
            self.abort()
        return False


def write_raster(path, data: np.ndarray, grid: Grid, times, channels, encoding: Encoding,
                 tags: dict | None = None, layout: str = "stack", **profile_overrides):
    """
    Write a (time, height, width, channel) array on ``grid``.
    Returns the path (``"stack"``) or the list of paths (``"per_time"``).
    """
    data = np.asarray(data)
    expected = (len(times), grid.height, grid.width, len(channels))
    if data.shape != expected:
        raise ValueError(f"array shape {data.shape} does not match (time, H, W, channel) "
                         f"{expected}")
    with RasterWriter(path, grid, times, channels, encoding, tags, layout,
                      **profile_overrides) as w:
        w.write(0, 0, data)
        paths = w.close()
    return paths[0] if layout == "stack" else paths


def _read_one(path):
    with rasterio.open(path) as src:
        tags = src.tags()
        arr = src.read()
        t = src.transform
        if abs(t.a) != abs(t.e) or t.b or t.d:
            raise ValueError(f"{path}: only north-up rasters with square pixels are supported")
        grid = Grid(src.crs, t.c, t.f, t.a, src.width, src.height)
        factor = float(tags.get("eetool_factor", 1.0))
        offset = float(tags.get("eetool_offset", 0.0))
        nodata = src.nodata if src.nodata is not None else (
            np.nan if np.issubdtype(arr.dtype, np.floating) else np.iinfo(arr.dtype).max)
        dtype = "float32" if np.issubdtype(arr.dtype, np.floating) else str(arr.dtype)
        encoding = Encoding(dtype=dtype, nodata=nodata, factor=factor, offset=offset)
        if "eetool_channels" in tags:
            channels = json.loads(tags["eetool_channels"])
            times = json.loads(tags.get("eetool_times", '["0"]'))
        else:  # any GeoTIFF: one time step, bands as channels
            channels = [d or f"b{i + 1}" for i, d in enumerate(src.descriptions)]
            times = ["0"]
    n_t, n_c = len(times), len(channels)
    if arr.shape[0] != n_t * n_c:
        raise ValueError(f"{path}: {arr.shape[0]} bands but {n_t} times x {n_c} channels")
    data = arr.reshape(n_t, n_c, *arr.shape[1:]).transpose(0, 2, 3, 1)
    meta = {k: v for k, v in tags.items()
            if not k.startswith("eetool_") or k == "eetool_product"}
    return np.ascontiguousarray(data), times, channels, grid, encoding, meta


def read_raster(path):
    """
    Read a GeoTIFF written by eetool (either layout) back into a
    :class:`eetool.RasterResult` with ``data`` of shape (time, H, W, channel).

    ``path`` is one file, or a list of per-time files (concatenated along
    time). Plain GeoTIFFs are read as one time step with bands as channels.
    """
    from .result import RasterResult

    paths = [path] if isinstance(path, (str, os.PathLike)) else list(path)
    if not paths:
        raise ValueError("No file given")
    parts = [_read_one(p) for p in paths]
    data, times, channels, grid, encoding, meta = parts[0]
    for d, t, c, g, e, _ in parts[1:]:
        if c != channels or g != grid or e != encoding:
            raise ValueError("Files differ in channels, grid or encoding")
    if len(parts) > 1:
        data = np.concatenate([p[0] for p in parts], axis=0)
        times = [t for p in parts for t in p[1]]
    meta = {k: v for k, v in meta.items() if k != "eetool_product"}
    product = parts[0][5].get("eetool_product", "")
    return RasterResult(data, times, channels, grid, encoding, product, meta)
