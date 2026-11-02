"""
Sentinel-1 GRD (COPERNICUS/S1_GRD, IW mode, backscatter in dB) composites
over consecutive time windows, one band per polarisation and window.

Storage: uint16 linear mapping of ``db_range`` (default -40..10 dB) onto
0..65534, NoData = 65535 (~0.0008 dB step). ``RasterResult.decode()`` or
:func:`dequantize_s1` gives float32 dB back.
"""

from __future__ import annotations

from typing import Sequence

from ..core import Encoding, Product, register, resolve_period, run, split_period
from ..core.grid import GridLike

COLLECTION = "COPERNICUS/S1_GRD"
POLARIZATIONS = ("VV", "VH", "HH", "HV")
REDUCERS = ("mean", "median", "min", "max")
DEFAULT_DB_RANGE = (-40.0, 10.0)
_LN10 = 2.302585092994046


def s1_encoding(db_range=DEFAULT_DB_RANGE) -> Encoding:
    return Encoding.from_range(float(db_range[0]), float(db_range[1]), dtype="uint16")


def quantize_s1(x_db, db_range=DEFAULT_DB_RANGE):
    """float dB -> uint16 (values outside db_range are clipped, NaN -> 65535)."""
    return s1_encoding(db_range).encode(x_db)


def dequantize_s1(q, db_range=DEFAULT_DB_RANGE):
    """uint16 -> float32 dB; 65535 -> NaN."""
    return s1_encoding(db_range).decode(q)


def _parse_orbit(orbit):
    if orbit is None:
        return None
    o = str(orbit).strip().upper()
    if o in ("BOTH", "ALL", "ANY", ""):
        return None
    if o in ("ASC", "ASCENDING", "A"):
        return "ASCENDING"
    if o in ("DESC", "DESCENDING", "D"):
        return "DESCENDING"
    raise ValueError(f"orbit must be 'ASCENDING', 'DESCENDING' or 'BOTH', got {orbit!r}")


@register
class S1Composite(Product):
    """Sentinel-1 GRD backscatter composites over time windows."""

    name = "s1_composite"
    default_res = 10

    def __init__(
        self,
        year: int | None = None,
        start_date=None,
        end_date=None,
        polarizations: Sequence[str] = ("VV",),
        orbit: str | None = "ASCENDING",
        window_days: int | None = 10,
        reducer: str = "mean",
        linear: bool = False,
        relative_orbits: Sequence[int] | None = None,
        platforms: Sequence[str] | None = None,
        mask_border_noise: bool = True,
        db_range: tuple[float, float] = DEFAULT_DB_RANGE,
    ):
        self.period = resolve_period(year, start_date, end_date)
        self.windows = split_period(self.period, window_days)
        self.window_days = window_days

        pols = [polarizations] if isinstance(polarizations, str) else list(polarizations)
        pols = list(dict.fromkeys(p.upper() for p in pols))
        bad = [p for p in pols if p not in POLARIZATIONS]
        if not pols or bad:
            raise ValueError(f"polarizations must be a subset of {POLARIZATIONS}, got {polarizations}")
        self.polarizations = pols

        self.orbit = _parse_orbit(orbit)
        if reducer not in REDUCERS:
            raise ValueError(f"reducer must be one of {REDUCERS}")
        self.reducer = reducer
        self.linear = bool(linear)
        self.relative_orbits = None if relative_orbits is None else [int(r) for r in (
            [relative_orbits] if isinstance(relative_orbits, int) else relative_orbits)]
        if platforms is None:
            self.platforms = None
        else:
            plats = [platforms] if isinstance(platforms, str) else list(platforms)
            # accept "A", "S1A", "Sentinel-1A"
            self.platforms = [str(p).upper().replace("SENTINEL-1", "").replace("S1", "")
                              for p in plats]
        self.mask_border_noise = mask_border_noise
        self.db_range = (float(db_range[0]), float(db_range[1]))
        self.encoding = s1_encoding(self.db_range)

    @property
    def band_names(self) -> list[str]:
        return [f"{p}_{w.start.isoformat()}" for p in self.polarizations for w in self.windows]

    def metadata(self) -> dict:
        return {
            "collection": COLLECTION,
            "polarizations": ",".join(self.polarizations),
            "orbit": self.orbit or "BOTH",
            "window_days": self.window_days if self.window_days else self.period.days,
            "reducer": self.reducer + (" (linear power)" if self.linear and self.reducer == "mean" else ""),
            "start_date": self.period.start.isoformat(),
            "end_date_exclusive": self.period.end.isoformat(),
            "units": "dB",
        }

    def collection(self, region):
        """Filtered (not yet windowed) collection with the selected polarisations."""
        import ee

        coll = (ee.ImageCollection(COLLECTION)
                .filterBounds(region)
                .filterDate(self.period.start.isoformat(), self.period.end.isoformat())
                .filter(ee.Filter.eq("instrumentMode", "IW")))
        for p in self.polarizations:
            coll = coll.filter(ee.Filter.listContains("transmitterReceiverPolarisation", p))
        if self.orbit:
            coll = coll.filter(ee.Filter.eq("orbitProperties_pass", self.orbit))
        if self.relative_orbits:
            coll = coll.filter(ee.Filter.inList("relativeOrbitNumber_start", self.relative_orbits))
        if self.platforms:
            coll = coll.filter(ee.Filter.inList("platform_number", self.platforms))
        return coll

    def _prep(self, image):
        img = image.select(self.polarizations)
        if self.mask_border_noise:
            # Low-value artefacts at swath edges (Earth Engine S1 guide)
            img = img.updateMask(img.gte(-30))
        if self.linear:
            img = img.multiply(0.1 * _LN10).exp()  # 10 ** (dB / 10), keeps band names
        return img.copyProperties(image, ["system:time_start"])

    def build_image(self, region):
        import ee

        coll = self.collection(region).map(lambda im: ee.Image(self._prep(im)))
        n_pol = len(self.polarizations)
        per_window = []
        for w in self.windows:
            sub = coll.filterDate(w.start.isoformat(), w.end.isoformat())
            names = [f"{p}_{w.start.isoformat()}" for p in self.polarizations]
            reduced = getattr(sub, self.reducer)()
            if self.linear:
                reduced = reduced.log10().multiply(10)
            empty = (ee.Image.constant([0] * n_pol).rename(names)
                     .updateMask(ee.Image.constant(0)))
            per_window.append(ee.Image(ee.Algorithms.If(
                sub.size().gt(0), reduced.select(self.polarizations, names), empty)))
        # window-major -> polarisation-major band order
        return ee.Image.cat(per_window).select(self.band_names)

    def prepare(self, grid, verbose=True):
        if not verbose:
            return
        import ee

        coll = self.collection(grid.ee_geometry())
        info = ee.Dictionary({
            "n": coll.size(),
            "pass": coll.aggregate_histogram("orbitProperties_pass"),
            "platform": coll.aggregate_histogram("platform_number"),
        }).getInfo()
        print(f"[eetool] {info['n']} Sentinel-1 scene(s); orbit pass {info['pass']}, "
              f"platform {info['platform']}; {len(self.windows)} window(s) x "
              f"{len(self.polarizations)} polarisation(s)")


def ee_download_s1_composite(
    raster_ref_image: GridLike,
    year: int | None = None,
    start_date=None,
    end_date=None,
    polarizations: Sequence[str] = ("VV",),
    orbit: str | None = "ASCENDING",
    window_days: int | None = 10,
    reducer: str = "mean",
    linear: bool = False,
    relative_orbits: Sequence[int] | None = None,
    platforms: Sequence[str] | None = None,
    mask_border_noise: bool = True,
    db_range: tuple[float, float] = DEFAULT_DB_RANGE,
    *,
    res: float | None = None,
    path_out=None,
    tile_size: int | None = None,
    n_workers: int = 4,
    max_retries: int = 5,
    verbose: bool = True,
):
    """
    Sentinel-1 GRD backscatter composites in consecutive ``window_days``
    windows (e.g. 10-day means over a year).

    Parameters
    ----------
    raster_ref_image : str | Path | eetool.Grid
        Reference raster (extent + CRS define the grid) or an :class:`eetool.Grid`.
    year : int, optional
        Shortcut for the calendar year; or give ``start_date`` / ``end_date``
        (ISO, ``end_date`` EXCLUSIVE).
    polarizations : sequence of str
        Any of "VV", "VH", "HH", "HV". Only scenes having all of them are used.
    orbit : {"ASCENDING", "DESCENDING", "BOTH"}
    window_days : int or None
        Window length; the last window is cut at the end date. ``None`` = one
        composite over the whole period.
    reducer : {"mean", "median", "min", "max"}
    linear : bool
        Average in linear power instead of dB (only changes "mean").
    relative_orbits : sequence of int, optional
        Keep only these relative orbit numbers (consistent viewing geometry).
    platforms : sequence of str, optional
        e.g. ``["A", "C"]`` for Sentinel-1A and 1C.
    mask_border_noise : bool
        Mask values below -30 dB (swath-edge noise) in every scene.
    db_range : (float, float)
        dB range mapped onto uint16 for storage.
    res, path_out, tile_size, n_workers, max_retries, verbose
        Download options shared by all eetool functions (default res: 10 m);
        see :func:`eetool.download`.

    Returns
    -------
    RasterResult | str
        ``path_out is None`` -> :class:`eetool.RasterResult`
        (n_polarizations * n_windows, H, W) uint16, bands named
        ``"VV_2024-01-01"``, ``"VV_2024-01-11"``, ... (polarisation-major);
        ``result.decode()`` -> float32 dB. Otherwise the GeoTIFF path.
    """
    product = S1Composite(
        year=year, start_date=start_date, end_date=end_date, polarizations=polarizations,
        orbit=orbit, window_days=window_days, reducer=reducer, linear=linear,
        relative_orbits=relative_orbits, platforms=platforms,
        mask_border_noise=mask_border_noise, db_range=db_range,
    )
    return run(product, raster_ref_image, res=res, path_out=path_out, tile_size=tile_size,
               n_workers=n_workers, max_retries=max_retries, verbose=verbose)
