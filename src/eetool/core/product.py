"""
Product base class and registry.

A *product* knows how to build one Earth Engine image (physical values,
masked where there is no data) for a region, how its output is laid out as
``times x channels`` and how the values are stored (:class:`Encoding`).

Every product produces a 4-D array (time, height, width, channel). Products
without a time dimension (percentile composites, the annual embedding) have
a single time step. On the Earth Engine side the image has
``len(times) * len(channels)`` bands in **time-major** order (all channels of
time 0, then all channels of time 1, ...).
Everything else — gridding, tiling, parallel ``computePixels`` requests,
retries, writing GeoTIFFs — is shared and lives in :mod:`eetool.core.download`.

Adding a new dataset::

    from eetool.core import Product, Encoding, register

    @register
    class MyProduct(Product):
        name = "my_product"
        default_res = 30

        def __init__(self, year):
            self.year = year
            self.encoding = Encoding.scaled(10000, "int16", -9999)

        @property
        def times(self):
            return [str(self.year)]

        @property
        def channels(self):
            return ["ndvi"]

        def build_image(self, region):
            ...  # ee.Image with len(times) * len(channels) bands, time-major
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from .encoding import Encoding
from .grid import Grid, Tile


class Product(ABC):
    #: registry key, used by ``eetool.download("<name>", ...)``
    name: ClassVar[str] = ""
    #: pixel size used when the caller passes ``res=None`` and the
    #: reference is a raster path (``None`` -> the reference's own pixel size)
    default_res: ClassVar[float | None] = None
    #: default tile size in pixels
    default_tile: ClassVar[int] = 256

    #: how values are stored; set as class attribute or in ``__init__``
    encoding: Encoding

    # ------------------------------------------------------------------ #
    # To implement
    # ------------------------------------------------------------------ #
    @property
    @abstractmethod
    def times(self) -> list[str]:
        """Labels of the time axis (e.g. ISO start date of each step)."""

    @property
    @abstractmethod
    def channels(self) -> list[str]:
        """Labels of the channel axis (e.g. spectral bands)."""

    @abstractmethod
    def build_image(self, region):
        """
        ``ee.Image`` with exactly ``len(times) * len(channels)`` bands in
        time-major order, holding physical (un-encoded) values, masked where
        there is no data. ``region`` is the ``ee.Geometry`` of the tile being
        requested — use it to ``filterBounds`` collections.
        """

    # ------------------------------------------------------------------ #
    # Optional hooks
    # ------------------------------------------------------------------ #
    def prepare(self, grid: Grid, verbose: bool = True) -> None:
        """Called once before downloading (validation, scene counts, ...)."""

    def metadata(self) -> dict:
        """Extra key/value tags written to the GeoTIFF."""
        return {}

    # ------------------------------------------------------------------ #
    # Derived
    # ------------------------------------------------------------------ #
    @property
    def shape(self) -> tuple[int, int]:
        """(n_times, n_channels)."""
        return len(self.times), len(self.channels)

    @property
    def band_names(self) -> list[str]:
        """Flat, time-major band names (used for 2-D GeoTIFF stacks)."""
        return flat_band_names(self.times, self.channels)

    # ------------------------------------------------------------------ #
    # Used by the download engine
    # ------------------------------------------------------------------ #
    def request_image(self, grid: Grid, tile: Tile, band_ids: list[str]):
        """Encoded image for one tile, bands renamed to ``band_ids``."""
        image = self.build_image(grid.ee_geometry(tile))
        return self.encoding.encode_ee(image).rename(band_ids)

    def __repr__(self) -> str:
        t, c = self.shape
        return f"<{type(self).__name__} '{self.name}': {t} time(s) x {c} channel(s)>"


def flat_band_names(times, channels) -> list[str]:
    """``channel`` when there is one time step, else ``"{channel}_{time}"`` (time-major)."""
    if len(times) == 1:
        return list(channels)
    return [f"{c}_{t}" for t in times for c in channels]


# ---------------------------------------------------------------------- #
# Registry
# ---------------------------------------------------------------------- #
PRODUCTS: dict[str, type[Product]] = {}


def register(cls: type[Product]) -> type[Product]:
    """Class decorator: make a product available by name."""
    if not cls.name:
        raise ValueError(f"{cls.__name__} needs a non-empty `name`")
    if cls.name in PRODUCTS and PRODUCTS[cls.name] is not cls:
        raise ValueError(f"A product named {cls.name!r} is already registered")
    PRODUCTS[cls.name] = cls
    return cls


def get_product(name: str) -> type[Product]:
    try:
        return PRODUCTS[name]
    except KeyError:
        raise KeyError(f"Unknown product {name!r}. Available: {sorted(PRODUCTS)}") from None


def list_products() -> list[str]:
    return sorted(PRODUCTS)
