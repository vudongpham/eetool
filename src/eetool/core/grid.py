"""
Output pixel grid: CRS + upper-left corner + pixel size + dimensions.

Every product in eetool is computed on a :class:`Grid`, so all products line
up pixel-for-pixel when they are built from the same reference.

A reference can be:

- a raster file (``.tif``, ``.vrt``, ...): its extent and CRS; ``res=None``
  keeps its pixel size;
- a vector file (``.gpkg``, ``.shp``, ``.geojson``, ``.fgb``, ...): the extent
  (xmin, ymin, xmax, ymax) of all its features; the grid starts exactly at
  (xmin, ymax);
- a GeoDataFrame / GeoSeries (anything with ``total_bounds`` and ``crs``);
- a ready-made :class:`Grid`.

Vectors in a geographic CRS (lon/lat) are put on the UTM zone of their centre
unless ``crs`` is given (see :meth:`Grid.from_vector`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import warnings
from typing import Any, NamedTuple, Union

import numpy as np
import rasterio
import rasterio.errors
from affine import Affine
from rasterio.crs import CRS
from rasterio.warp import transform_bounds

#: file extensions read as vectors (anything else is tried as a raster first)
VECTOR_EXTENSIONS = {".gpkg", ".shp", ".geojson", ".json", ".fgb", ".gml", ".kml",
                     ".kmz", ".gdb", ".parquet", ".geoparquet", ".tab", ".mif", ".sqlite",
                     ".gpx", ".dxf", ".csv", ".zip"}
RASTER_EXTENSIONS = {".tif", ".tiff", ".vrt", ".img", ".jp2", ".nc", ".hdf", ".h5",
                     ".asc", ".dem", ".bil", ".png", ".jpg", ".jpeg"}


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

    @classmethod
    def from_vector(cls, source, res: float, crs=None, layer=None, snap: bool = False) -> "Grid":
        """
        Grid clipped to the extent (xmin, ymin, xmax, ymax) of all features of
        a vector file (``.gpkg``, ``.shp``, ``.geojson``, ...) or of a
        GeoDataFrame / GeoSeries. The upper-left pixel corner is exactly
        (xmin, ymax); the width/height are rounded up to whole pixels, so the
        last column/row can reach up to one pixel past xmax / below ymin.

        Parameters
        ----------
        source : path or GeoDataFrame / GeoSeries
        res : float
            Pixel size in units of the output CRS (vectors have no pixel size).
        crs : optional
            Output CRS (e.g. ``"EPSG:32633"``); the bounds are reprojected.
            Default: the vector's CRS, or, if that is geographic (lon/lat),
            the UTM zone of the extent's centre.
        layer : str | int, optional
            Layer of a multi-layer file (e.g. a GeoPackage); default the first.
        snap : bool
            Default False. True = snap the extent outward to multiples of
            ``res`` so grids built from different vectors line up.
        """
        if res is None or res <= 0:
            raise ValueError("A vector reference needs a pixel size: pass `res` (e.g. res=30)")
        src_crs, bounds = _vector_crs_bounds(source, layer)
        if src_crs is None:
            raise ValueError(f"Vector reference {source!r} has no CRS")
        if crs is None and src_crs.is_geographic:
            crs = utm_crs_for(bounds, src_crs)
            warnings.warn(
                f"The vector reference is in a geographic CRS ({src_crs.to_string()}); "
                f"using {crs.to_string()} (UTM zone of its centre). Pass "
                "`eetool.Grid.from_vector(path, res, crs=...)` to choose another CRS.",
                stacklevel=3,
            )
        crs = src_crs if crs is None else CRS.from_user_input(crs)
        if crs != src_crs:
            bounds = transform_bounds(src_crs, crs, *bounds, densify_pts=21)
        left, bottom, right, top = bounds
        if right <= left or top <= bottom:  # a single point / a line along an axis
            left, right = (left, left + res) if right <= left else (left, right)
            bottom, top = (top - res, top) if top <= bottom else (bottom, top)
        if snap:
            left, bottom = np.floor(left / res) * res, np.floor(bottom / res) * res
            right, top = np.ceil(right / res) * res, np.ceil(top / res) * res
        return cls.from_bounds((left, bottom, right, top), crs, res)

    @classmethod
    def from_reference(cls, reference, res: float | None = None) -> "Grid":
        """Raster file, vector file or GeoDataFrame -> Grid (see module docs)."""
        if isinstance(reference, Grid):
            if res is not None and not np.isclose(res, reference.res):
                raise ValueError(f"res={res} conflicts with the Grid's res={reference.res}")
            return reference
        if _is_geodataframe(reference):
            return cls.from_vector(reference, res)
        if not isinstance(reference, (str, Path)):
            raise TypeError("reference must be a raster/vector path, a GeoDataFrame/GeoSeries "
                            f"or an eetool.Grid, got {type(reference).__name__}")
        kind = reference_kind(reference)
        if kind == "vector":
            return cls.from_vector(reference, res)
        return cls.from_raster(reference, res)

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


GridLike = Union[str, Path, Grid, Any]


def _is_geodataframe(obj) -> bool:
    return hasattr(obj, "total_bounds") and hasattr(obj, "crs")


def _vector_crs_bounds(source, layer=None):
    if _is_geodataframe(source):
        crs = None if source.crs is None else CRS.from_user_input(source.crs)
        bounds = tuple(float(v) for v in source.total_bounds)
    else:
        try:
            import pyogrio
        except ImportError:  # pragma: no cover
            raise ImportError("Reading vector references needs pyogrio: pip install pyogrio") from None
        info = pyogrio.read_info(str(source), layer=layer, force_total_bounds=True)
        crs = None if not info.get("crs") else CRS.from_user_input(info["crs"])
        bounds = info.get("total_bounds")
        if bounds is None or info.get("features", 1) == 0:
            raise ValueError(f"Vector reference {source} has no features")
        bounds = tuple(float(v) for v in bounds)
    if not np.all(np.isfinite(bounds)):
        raise ValueError(f"Vector reference {source!r} has no valid geometries")
    return crs, bounds


def utm_crs_for(bounds, crs) -> CRS:
    """WGS 84 / UTM zone (EPSG:326xx north, 327xx south) of the centre of ``bounds``."""
    lon0, lat0, lon1, lat1 = transform_bounds(crs, "EPSG:4326", *bounds)
    lon, lat = (lon0 + lon1) / 2, (lat0 + lat1) / 2
    zone = int(np.clip(np.floor((lon + 180) / 6) + 1, 1, 60))
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


def reference_kind(path) -> str:
    """``"raster"`` or ``"vector"`` for a reference file."""
    ext = Path(path).suffix.lower()
    if ext in VECTOR_EXTENSIONS:
        return "vector"
    if ext in RASTER_EXTENSIONS:
        return "raster"
    try:  # unknown extension: ask GDAL through rasterio first
        with rasterio.open(path):
            return "raster"
    except rasterio.errors.RasterioIOError:
        return "vector"


def resolve_grid(reference: GridLike, res: float | None) -> Grid:
    """Raster path, vector path, GeoDataFrame or :class:`Grid` -> :class:`Grid`."""
    return Grid.from_reference(reference, res)
