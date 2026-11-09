"""In-memory result: a 4-D (time, height, width, channel) array plus its grid."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .encoding import Encoding
from .grid import Grid
from .io import write_raster


@dataclass
class RasterResult:
    """
    Encoded pixels plus everything needed to interpret or save them.

    ``data`` has shape **(time, height, width, channel)** and the product's
    encoded dtype (NoData = ``nodata``). Products without a time dimension
    have ``time = 1``. ``np.asarray(result)``, ``result.shape`` and
    ``result[...]`` work directly on ``data``.

    Georeferencing for writing it yourself: ``result.transform`` (affine,
    upper-left corner), ``result.crs``, ``result.nodata``.
    """

    data: np.ndarray
    times: list[str]
    channels: list[str]
    grid: Grid
    encoding: Encoding
    product: str = ""
    tags: dict = field(default_factory=dict)

    def __post_init__(self):
        self.times, self.channels = list(self.times), list(self.channels)
        expected = (len(self.times), self.grid.height, self.grid.width, len(self.channels))
        if self.data.shape != expected:
            raise ValueError(f"data shape {self.data.shape} != (time, H, W, channel) {expected}")

    # array-like behaviour ------------------------------------------------
    def __array__(self, dtype=None, copy=None):
        return self.data if dtype is None else self.data.astype(dtype)

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def nodata(self):
        return self.encoding.nodata

    @property
    def crs(self):
        return self.grid.crs

    @property
    def transform(self):
        return self.grid.transform

    @property
    def band_names(self) -> list[str]:
        """Flat time-major names, as used in a stacked GeoTIFF."""
        from .product import flat_band_names

        return flat_band_names(self.times, self.channels)

    # helpers ---------------------------------------------------------------
    def decode(self) -> np.ndarray:
        """float32 physical values (time, H, W, channel), NaN where NoData."""
        return self.encoding.decode(self.data)

    def valid(self) -> np.ndarray:
        """Boolean (time, H, W, channel): True where there is data."""
        return self.data != self.encoding.nodata

    def _index(self, labels, values, axis):
        if values is None:
            return list(range(len(labels)))
        values = [values] if isinstance(values, (str, int)) else list(values)
        out = []
        for v in values:
            if isinstance(v, int):
                out.append(v)
            elif v in labels:
                out.append(labels.index(v))
            else:
                raise KeyError(f"No {axis} {v!r}. Available: {labels}")
        return out

    def sel(self, time=None, channel=None) -> "RasterResult":
        """
        Subset by label or position, keeping the 4-D layout, e.g.
        ``res.sel(channel=["red", "nir"])`` or ``res.sel(time="2015-01-08")``.
        """
        ti = self._index(self.times, time, "time")
        ci = self._index(self.channels, channel, "channel")
        data = self.data[ti][..., ci]
        return RasterResult(data, [self.times[i] for i in ti], [self.channels[i] for i in ci],
                            self.grid, self.encoding, self.product, dict(self.tags))

    def to_bands(self) -> np.ndarray:
        """(time * channel, H, W) band stack in time-major order (rasterio layout)."""
        t, h, w, c = self.data.shape
        return self.data.transpose(0, 3, 1, 2).reshape(t * c, h, w)

    def to_raster(self, path_out, layout: str = "stack"):
        """
        Write GeoTIFF(s) on the download grid.

        ``layout="stack"`` -> one file with time * channel bands (time-major);
        ``layout="per_time"`` -> one file per time step, ``<stem>_<time>.tif``.
        Returns the path, or the list of paths. :func:`eetool.read_raster`
        rebuilds the 4-D array from either.
        """
        return write_raster(path_out, self.data, self.grid, self.times, self.channels,
                            self.encoding, {"eetool_product": self.product, **self.tags},
                            layout=layout)

    def to_xarray(self, decode: bool = False):
        """``xarray.DataArray`` with dims (time, y, x, channel); needs xarray."""
        import xarray as xr

        t = self.grid.transform
        x = t.c + t.a * (np.arange(self.grid.width) + 0.5)
        y = t.f + t.e * (np.arange(self.grid.height) + 0.5)
        data = self.decode() if decode else self.data
        attrs = {"crs": self.grid.crs.to_wkt(), "transform": tuple(t)[:6],
                 "product": self.product, **{k: str(v) for k, v in self.tags.items()}}
        if not decode:
            attrs.update(nodata=self.nodata, scale_factor=1 / self.encoding.factor,
                         add_offset=self.encoding.offset)
        return xr.DataArray(data, dims=("time", "y", "x", "channel"),
                            coords={"time": self.times, "y": y, "x": x,
                                    "channel": self.channels},
                            name=self.product or "eetool", attrs=attrs)

    def __repr__(self) -> str:
        return (f"RasterResult(product={self.product!r}, shape={self.shape} "
                f"(time, H, W, channel), dtype={self.data.dtype}, nodata={self.nodata}, "
                f"crs={self.grid.crs})")
