"""
Generic tiled downloader shared by every product.

The grid is split into tiles; each tile is fetched with
``ee.data.computePixels`` (exact output grid, no reprojection guesswork),
in parallel, with retries. Each tile comes back as a (time * channel, h, w)
band stack and is reshaped to (time, h, w, channel). Tiles are assembled in
memory (-> :class:`RasterResult` with a 4-D array) or streamed straight into
GeoTIFF(s), see :mod:`eetool.core.io` for the file layouts.
"""

from __future__ import annotations

import os
import random
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Union

import numpy as np

from .grid import Grid, GridLike, resolve_grid
from .io import LAYOUTS, RasterWriter
from .product import Product, get_product
from .result import RasterResult

#: Earth Engine's computePixels payload limit is 48 MB; keep a margin.
MAX_REQUEST_BYTES = 40 * 1024 ** 2

# Error messages that will not go away by retrying.
_NON_RETRYABLE = (
    "not found",
    "did not match any bands",
    "does not have",
    "invalid argument",
    "unrecognized argument",
    "permission",
    "not authorized",
    "please authorize",
    "not been initialized",
    "no project found",
)

DOWNLOAD_OPTIONS = ("res", "path_out", "layout", "tile_size", "n_workers", "max_retries",
                    "showProgress")


def _is_retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return not any(s in msg for s in _NON_RETRYABLE)


def fetch_pixels(image, band_ids, grid_request, max_retries=5):
    """One computePixels call -> (bands, h, w) array, with exponential back-off."""
    import ee

    request = {
        "expression": image,
        "fileFormat": "NUMPY_NDARRAY",
        "bandIds": list(band_ids),
        "grid": grid_request,
    }
    for attempt in range(max_retries + 1):
        try:
            arr = ee.data.computePixels(request)
            # structured array (h, w) with one field per band -> (bands, h, w)
            return np.stack([arr[b] for b in band_ids], axis=0)
        except (ee.EEException, OSError, TimeoutError) as exc:
            if attempt == max_retries or not _is_retryable(exc):
                raise
            time.sleep(min(2 ** attempt, 60) * (0.75 + 0.5 * random.random()))


def _safe_tile_size(requested: int, n_bands: int, itemsize: int) -> int:
    limit = int(np.sqrt(MAX_REQUEST_BYTES / (n_bands * itemsize)))
    if requested > limit:
        size = max(16, limit // 16 * 16)
        warnings.warn(
            f"tile_size={requested} with {n_bands} bands exceeds the Earth Engine "
            f"request limit; using tile_size={size}.", stacklevel=3,
        )
        return size
    return requested


def _fmt_seconds(s: float) -> str:
    s = int(round(s))
    return f"{s // 3600:d}:{s % 3600 // 60:02d}:{s % 60:02d}" if s >= 3600 else f"{s // 60:02d}:{s % 60:02d}"


def _progress(done, total, t0, width=30):
    """One-line progress bar, redrawn in place: [#####-----]  12/40 tiles  30%  00:12<00:28"""
    elapsed = time.time() - t0
    frac = done / total if total else 1.0
    eta = elapsed / done * (total - done) if done else 0.0
    bar = "#" * int(round(frac * width))
    sys.stdout.write(f"\r[eetool] [{bar:-<{width}}] {done:>{len(str(total))}}/{total} tiles "
                     f"{frac:4.0%}  {_fmt_seconds(elapsed)}<{_fmt_seconds(eta)}")
    sys.stdout.flush()


def run(
    product: Product,
    reference: GridLike,
    *,
    res: float | None = None,
    path_out: str | os.PathLike | None = None,
    layout: str = "stack",
    tile_size: int | None = None,
    n_workers: int = 4,
    max_retries: int = 5,
    showProgress: bool = False,
) -> Union[RasterResult, str, list]:
    """
    Download ``product`` on the grid of ``reference``.

    Returns a :class:`RasterResult` whose ``data`` is (time, H, W, channel)
    when ``path_out`` is None. Otherwise streams the tiles to disk without
    holding the whole array in memory and returns the GeoTIFF path
    (``layout="stack"``) or list of paths (``layout="per_time"``).

    ``showProgress=True`` prints the grid size, scene counts and a progress
    bar of the tiles received from Earth Engine; ``False`` (default) prints
    nothing and skips the extra scene-count queries.
    """
    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    if res is None and not isinstance(reference, Grid):
        res = product.default_res
    grid = resolve_grid(reference, res)

    times, channels = list(product.times), list(product.channels)
    n_t, n_c = len(times), len(channels)
    n_bands = n_t * n_c
    if n_bands == 0:
        raise ValueError(f"{product!r} has no output bands")
    band_ids = [f"b{i:04d}" for i in range(n_bands)]
    itemsize = np.dtype(product.encoding.dtype).itemsize
    tile_size = _safe_tile_size(tile_size or product.default_tile, n_bands, itemsize)
    tiles = grid.tiles(tile_size)

    if showProgress:
        size_mb = grid.width * grid.height * n_bands * itemsize / 1024 ** 2
        print(f"[eetool] {product.name}: {n_t} time(s) x {n_c} channel(s), grid "
              f"{grid.width} x {grid.height} px @ {grid.res:g}, "
              f"{len(tiles)} tiles, ~{size_mb:,.0f} MB {product.encoding.dtype}")

    product.prepare(grid, showProgress)

    tags = {"eetool_product": product.name, **product.metadata()}

    def work(tile):
        image = product.request_image(grid, tile, band_ids)
        try:
            data = fetch_pixels(image, band_ids, grid.tile_grid_request(tile), max_retries)
        except Exception as exc:
            raise RuntimeError(f"Tile {tile} failed: {exc}") from exc
        # (time * channel, h, w), time-major -> (time, h, w, channel)
        return tile, data.reshape(n_t, n_c, tile.height, tile.width).transpose(0, 2, 3, 1)

    def iterate():
        t0 = time.time()
        if showProgress:
            _progress(0, len(tiles), t0)
        with ThreadPoolExecutor(max_workers=max(1, int(n_workers))) as pool:
            futures = [pool.submit(work, t) for t in tiles]
            try:
                for done, fut in enumerate(as_completed(futures), start=1):
                    yield fut.result()
                    if showProgress:
                        _progress(done, len(tiles), t0)
            except BaseException:
                for f in futures:
                    f.cancel()
                raise
            finally:
                if showProgress:
                    print()

    if path_out is None:
        out = np.full((n_t, grid.height, grid.width, n_c), product.encoding.nodata,
                      dtype=product.encoding.dtype)
        for t, data in iterate():
            out[:, t.row:t.row + t.height, t.col:t.col + t.width, :] = data
        return RasterResult(out, times, channels, grid, product.encoding, product.name, tags)

    with RasterWriter(path_out, grid, times, channels, product.encoding, tags,
                      layout=layout) as writer:
        for t, data in iterate():
            writer.write(t.row, t.col, data)
        paths = writer.close()
    if showProgress:
        print(f"[eetool] wrote {paths[0]}" if len(paths) == 1
              else f"[eetool] wrote {len(paths)} files: {paths[0]} ... {paths[-1]}")
    return paths[0] if layout == "stack" else paths


def get(product, reference: GridLike, **kwargs):
    """
    Get any registered product by name (or a :class:`Product` instance).

    Keyword arguments named in ``DOWNLOAD_OPTIONS`` (``res``, ``path_out``,
    ``layout``, ``tile_size``, ``n_workers``, ``max_retries``, ``showProgress``) control the
    download; all others are passed to the product's constructor::

        eetool.get("s1_composite", "ref.tif", start_date="20240101",
                        end_date="20241231",
                        polarizations=["VV", "VH"], path_out="s1.tif")
    """
    options = {k: kwargs.pop(k) for k in list(kwargs) if k in DOWNLOAD_OPTIONS}
    if isinstance(product, str):
        product = get_product(product)(**kwargs)
    elif kwargs:
        raise TypeError(f"Unexpected arguments for a Product instance: {sorted(kwargs)}")
    if not isinstance(product, Product):
        raise TypeError("product must be a registered name or a Product instance")
    return run(product, reference, **options)


#: old name of :func:`get`
download = get
