"""
Output pixel grid: CRS + upper-left corner + pixel size + dimensions.

Every download in eetool is defined on a :class:`Grid`, so all products line
up pixel-for-pixel when they are built from the same reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, NamedTuple, Union

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS


class Tile(NamedTuple):
    """A block of the grid in pixel space (row/col of its upper-left pixel)."""

    row: int
    col: int
    height: int
    width: int


@dataclass(frozen=True)
class Grid:
    crs: CRS
    x_ul: float
    y_ul: float
    res: float
    width: int
    height: int

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_bounds(cls, bounds, crs, res: float) -> "Grid":
        """
        Grid anchored at the upper-left corner of ``bounds``
        (left, bottom, right, top) with square pixels of size ``res``
        (CRS units). The extent is rounded *up* to whole pixels.
        """
        crs = CRS.from_user_input(crs)
        left, bottom, right, top = (float(v) for v in bounds)
        if res is None or res <= 0:
            raise ValueError("res must be a positive number")
        if right <= left or top <= bottom:
            raise ValueError(f"Invalid bounds {bounds}")
        if crs.is_geographic and res > 1:
            raise ValueError(
                f"The CRS is geographic (degrees) but res={res} looks like metres. "
                "Reproject the reference to a projected CRS, or pass res in degrees."
            )
        width = int(np.ceil((right - left) / res - 1e-6))
        height = int(np.ceil((top - bottom) / res - 1e-6))
        return cls(crs=crs, x_ul=left, y_ul=top, res=float(res),
                   width=width, height=height)

    @classmethod
    def from_raster(cls, path, res: float | None = None) -> "Grid":
        """
        Grid from a reference raster's extent and CRS. ``res=None`` keeps the
        reference raster's own (square) pixel size.
        """
        with rasterio.open(path) as src:
            crs, bounds = src.crs, src.bounds
            native = src.res
        if crs is None:
            raise ValueError(f"Reference raster {path} has no CRS")
        if res is None:
            if not np.isclose(native[0], native[1]):
                raise ValueError(f"Reference pixels are not square {native}; pass `res`.")
            res = native[0]
        return cls.from_bounds(bounds, crs, res)

    # ------------------------------------------------------------------ #
    @property
    def transform(self) -> Affine:
        return Affine(self.res, 0.0, self.x_ul, 0.0, -self.res, self.y_ul)

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return (self.x_ul, self.y_ul - self.height * self.res,
                self.x_ul + self.width * self.res, self.y_ul)

    @property
    def ee_crs(self) -> str:
        """CRS string accepted by ``ee.Projection`` (EPSG code if possible)."""
        epsg = self.crs.to_epsg()
        return f"EPSG:{epsg}" if epsg is not None else self.crs.to_wkt()

    @property
    def _crs_field(self) -> dict:
        epsg = self.crs.to_epsg()
        return {"crsCode": f"EPSG:{epsg}"} if epsg is not None else {"crsWkt": self.crs.to_wkt()}

    # ------------------------------------------------------------------ #
    # Tiling
    # ------------------------------------------------------------------ #
    def tiles(self, size: int) -> list[Tile]:
        """Non-overlapping tiles of at most ``size`` x ``size`` pixels."""
        if size < 1:
            raise ValueError("tile size must be >= 1")
        return [
            Tile(r, c, min(size, self.height - r), min(size, self.width - c))
            for r in range(0, self.height, size)
            for c in range(0, self.width, size)
        ]

    def tile_bounds(self, tile: Tile) -> tuple[float, float, float, float]:
        x0 = self.x_ul + tile.col * self.res
        y1 = self.y_ul - tile.row * self.res
        return (x0, y1 - tile.height * self.res, x0 + tile.width * self.res, y1)

    def tile_grid_request(self, tile: Tile) -> dict:
        """The ``grid`` part of an ``ee.data.computePixels`` request."""
        x0, _, _, y1 = self.tile_bounds(tile)
        return {
            "dimensions": {"width": int(tile.width), "height": int(tile.height)},
            "affineTransform": {
                "scaleX": self.res, "shearX": 0, "translateX": x0,
                "shearY": 0, "scaleY": -self.res, "translateY": y1,
            },
            **self._crs_field,
        }

    # ------------------------------------------------------------------ #
    # Earth Engine geometries (need an initialised ee session)
    # ------------------------------------------------------------------ #
    def ee_geometry(self, tile: Tile | None = None):
        """``ee.Geometry.Rectangle`` of the whole grid or of one tile."""
        import ee

        b = self.bounds if tile is None else self.tile_bounds(tile)
        return ee.Geometry.Rectangle(list(b), proj=ee.Projection(self.ee_crs),
                                     geodesic=False)


GridLike = Union[str, Path, Grid]


def resolve_grid(ref: GridLike, res: float | None) -> Grid:
    """Accept a raster path or a ready-made :class:`Grid`."""
    if isinstance(ref, Grid):
        if res is not None and not np.isclose(res, ref.res):
            raise ValueError(f"res={res} conflicts with the Grid's res={ref.res}")
        return ref
    return Grid.from_raster(ref, res)
