"""In-memory download result."""

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

    ``data`` has shape (bands, height, width) and the product's encoded dtype.
    The object behaves like that array for ``np.asarray(result)``,
    ``result.shape`` and indexing.
    """

    data: np.ndarray
    band_names: list[str]
    grid: Grid
    encoding: Encoding
    product: str = ""
    tags: dict = field(default_factory=dict)

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

    # helpers ---------------------------------------------------------------
    def decode(self) -> np.ndarray:
        """float32 physical values, NaN where NoData."""
        return self.encoding.decode(self.data)

    def band(self, name: str, decode: bool = False) -> np.ndarray:
        """One band by name, e.g. ``result.band("p50_nir")``."""
        try:
            i = self.band_names.index(name)
        except ValueError:
            raise KeyError(f"No band {name!r}. Bands: {self.band_names}") from None
        arr = self.data[i]
        return self.encoding.decode(arr) if decode else arr

    def select(self, names) -> "RasterResult":
        """Subset of bands (keeps order given)."""
        idx = []
        for n in names:
            if n not in self.band_names:
                raise KeyError(f"No band {n!r}")
            idx.append(self.band_names.index(n))
        return RasterResult(self.data[idx], [self.band_names[i] for i in idx],
                            self.grid, self.encoding, self.product, dict(self.tags))

    def to_raster(self, path_out) -> str:
        """Write a GeoTIFF on the download grid; returns the path."""
        return write_raster(path_out, self.data, self.grid, self.band_names,
                            self.encoding, {"eetool_product": self.product, **self.tags})

    def __repr__(self) -> str:
        return (f"RasterResult(product={self.product!r}, shape={self.shape}, "
                f"dtype={self.data.dtype}, nodata={self.nodata}, crs={self.grid.crs})")
