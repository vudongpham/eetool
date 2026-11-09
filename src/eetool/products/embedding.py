"""
Google Satellite Embedding (GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL): 64 bands
A00..A63, unit-length vectors with values in [-1, 1], one image set per year.

Storage: uint16, q = round((x + 1) / 2 * 65534) -> 0..65534, NoData = 65535.
Max quantisation error ~1.5e-5. ``RasterResult.decode()`` or
:func:`dequantize_embedding` gives float32 back.
"""

from __future__ import annotations

from typing import Sequence

from ..core import Encoding, Product, register, run
from ..core.grid import GridLike

COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
EMBED_BANDS = [f"A{i:02d}" for i in range(64)]
ENCODING = Encoding.from_range(-1.0, 1.0, dtype="uint16")


def quantize_embedding(x):
    """float in [-1, 1] -> uint16 in [0, 65534] (NaN -> 65535)."""
    return ENCODING.encode(x)


def dequantize_embedding(q):
    """uint16 -> float32 in [-1, 1]; 65535 -> NaN."""
    return ENCODING.decode(q)


@register
class AnnualEmbedding(Product):
    """Annual 64-D Google Satellite Embedding."""

    name = "annual_embedding"
    default_res = 10
    encoding = ENCODING

    def __init__(self, year: int, bands: Sequence[str] | None = None):
        self.year = int(year)
        if bands is None:
            self.bands = list(EMBED_BANDS)
        else:
            bands = [bands] if isinstance(bands, str) else list(bands)
            bad = [b for b in bands if b not in EMBED_BANDS]
            if bad:
                raise ValueError(f"Unknown embedding band(s) {bad}; use A00..A63")
            self.bands = bands

    @property
    def times(self) -> list[str]:
        return [str(self.year)]

    @property
    def channels(self) -> list[str]:
        return list(self.bands)

    def metadata(self) -> dict:
        return {"collection": COLLECTION, "year": self.year}

    def collection(self, region):
        import ee

        return (ee.ImageCollection(COLLECTION)
                .filterDate(f"{self.year}-01-01", f"{self.year + 1}-01-01")
                .filterBounds(region))

    def build_image(self, region):
        # Embedding tiles are in local UTM zones; mosaic() + computePixels warps
        # them onto the grid with nearest neighbour (vectors are never blended).
        return self.collection(region).mosaic().select(self.bands)

    def prepare(self, grid, verbose=True):
        n = self.collection(grid.ee_geometry()).size().getInfo()
        if n == 0:
            raise ValueError(f"No Satellite Embedding images for {self.year} over this "
                             "extent (check that the year is covered by the dataset).")
        if verbose:
            print(f"[eetool] {n} embedding image(s) for {self.year}")


def ee_get_annual_embedding(
    reference: GridLike,
    year: int,
    bands: Sequence[str] | None = None,
    *,
    res: float | None = None,
    path_out=None,
    layout: str = "stack",
    tile_size: int | None = None,
    n_workers: int = 4,
    max_retries: int = 5,
    showProgress: bool = False,
):
    """
    Annual Google Satellite Embedding on the reference grid.

    Parameters
    ----------
    reference : str | Path | GeoDataFrame | eetool.Grid
        Defines the output grid: a raster file (its extent and CRS), a vector
        file (``.gpkg``, ``.shp``, ``.geojson``, ...; the extent of all its
        features), a GeoDataFrame / GeoSeries, or an :class:`eetool.Grid`.
    year : int
        Embedding year.
    bands : sequence of str, optional
        Subset of ``A00`` .. ``A63`` (default: all 64).
    res, path_out, layout, tile_size, n_workers, max_retries, showProgress
        Options shared by all eetool functions (default res: 10 m);
        see :func:`eetool.get`.

    Returns
    -------
    RasterResult | str | list[str]
        ``path_out is None`` -> :class:`eetool.RasterResult` with ``data`` of
        shape (1, H, W, n_bands) uint16 (time = the year; 128 bytes per pixel
        for all 64 bands — stream large areas with ``path_out``).
        ``result.decode()`` -> float32 in [-1, 1]. Otherwise the GeoTIFF path(s).
    """
    product = AnnualEmbedding(year=year, bands=bands)
    return run(product, reference, res=res, path_out=path_out, layout=layout,
               tile_size=tile_size, n_workers=n_workers, max_retries=max_retries,
               showProgress=showProgress)
