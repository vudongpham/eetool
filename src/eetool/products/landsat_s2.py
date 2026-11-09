"""
Cloud-masked Landsat 4/5/7/8/9 Collection 2 Level-2 and/or Sentinel-2 L2A
(harmonized; any platform A, B, C, D, ...) surface reflectance:

- ``LandsatS2STM``: spectral-temporal metrics (per-pixel percentiles over the
  whole period).
- ``LandsatS2TE``: temporal encoding (one composite per ``time_step`` days).

Storage: reflectance * 10000 as int16, NoData = -9999.
"""

from __future__ import annotations

import re
import warnings
from datetime import timedelta
from typing import Sequence

import numpy as np

from ..core import Encoding, Product, register, resolve_period, run, split_period
from ..core.grid import GridLike

COMMON_BANDS = ["blue", "green", "red", "nir", "swir1", "swir2"]

# Landsat Collection 2 Level-2 surface reflectance
_TM_ETM_BANDS = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"]
_OLI_BANDS = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]
LANDSAT_COLLECTIONS = {
    "L4": ("LANDSAT/LT04/C02/T1_L2", _TM_ETM_BANDS),
    "L5": ("LANDSAT/LT05/C02/T1_L2", _TM_ETM_BANDS),
    "L7": ("LANDSAT/LE07/C02/T1_L2", _TM_ETM_BANDS),
    "L8": ("LANDSAT/LC08/C02/T1_L2", _OLI_BANDS),
    "L9": ("LANDSAT/LC09/C02/T1_L2", _OLI_BANDS),
}

# Sentinel-2 L2A (harmonized: post-2022 baseline offset already removed)
S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
S2_CLOUD_SCORE = "GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED"

ENCODING = Encoding.scaled(10000, dtype="int16", nodata=-9999)
TE_REDUCERS = ("mean", "median", "min", "max")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def percentile_label(p: float) -> str:
    """0.1 -> 'p10', 0.025 -> 'p2.5', 0.5 -> 'p50'."""
    return f"p{round(p * 100, 6):g}"


def normalize_percentiles(percentiles) -> list[float]:
    """Validate fractions in [0, 1]; keep the user's order."""
    p = np.atleast_1d(np.asarray(percentiles, dtype=float))
    if p.ndim != 1 or p.size == 0:
        raise ValueError("percentiles must be a non-empty 1-D list, e.g. [0.1, 0.5, 0.75]")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("percentiles must be fractions in [0, 1] (e.g. 0.1 for the "
                         f"10th percentile), got {p.tolist()}")
    if len(np.unique(p)) != len(p):
        raise ValueError(f"percentiles contain duplicates: {p.tolist()}")
    return [float(x) for x in p]


def parse_sensors(sensors):
    """
    Accepted names (case-insensitive; '-', '_' and spaces ignored):
        Landsat:    L4, L5, L7, L8, L9   (or LANDSAT8, Landsat-9, ...)
        Sentinel-2: S2A, S2B, S2C, ...   (or SENTINEL2C, Sentinel-2B, ...)
                    S2 = every Sentinel-2 platform
    Returns (landsat_keys, s2_platforms); s2_platforms is None if no S2 was
    requested, [] for "all platforms", else a list of SPACECRAFT_NAME values.
    """
    if isinstance(sensors, str):
        sensors = [sensors]
    landsat, s2_platforms, s2_all = [], [], False
    for s in sensors:
        key = re.sub(r"[\s_\-]", "", str(s)).upper()
        key = re.sub(r"^LANDSAT", "L", key)
        key = re.sub(r"^SENTINEL2", "S2", key)
        if key in LANDSAT_COLLECTIONS:
            if key not in landsat:
                landsat.append(key)
        elif key == "S2":
            s2_all = True
        elif re.fullmatch(r"S2[A-Z]", key):
            name = f"Sentinel-2{key[-1]}"
            if name not in s2_platforms:
                s2_platforms.append(name)
        else:
            raise ValueError(f"Unknown sensor {s!r}. Use L4, L5, L7, L8, L9, S2, or S2A/S2B/S2C/...")
    s2 = [] if s2_all else (s2_platforms or None)
    if not landsat and s2 is None:
        raise ValueError("No sensors selected.")
    return landsat, s2


def _parse_bands(bands):
    bands = list(COMMON_BANDS if bands is None else ([bands] if isinstance(bands, str) else bands))
    bad = [b for b in bands if b not in COMMON_BANDS]
    if bad:
        raise ValueError(f"Unknown band(s) {bad}; choose from {COMMON_BANDS}")
    if not bands:
        raise ValueError("Select at least one band")
    return list(dict.fromkeys(bands))


# ---------------------------------------------------------------------------
# Product
# ---------------------------------------------------------------------------
class _LandsatS2Base(Product):
    """Shared scene selection, cloud masking and scaling for Landsat / S2."""

    default_res = 30
    encoding = ENCODING

    def __init__(
        self,
        start_date,
        end_date,
        sensors: Sequence[str] = ("L4", "L5", "L7", "L8", "L9"),
        bands: Sequence[str] | None = None,
        max_scene_cloud: float | None = None,
        s2_nir: str = "B8A",
        s2_cs_threshold: float = 0.6,
        mask_snow: bool = True,
        resample: str | None = None,
    ):
        self.period = resolve_period(start_date, end_date)
        self.landsat_keys, self.s2_platforms = parse_sensors(sensors)
        self.bands = _parse_bands(bands)
        if s2_nir not in ("B8A", "B8"):
            raise ValueError("s2_nir must be 'B8A' or 'B8'")
        if not 0 <= s2_cs_threshold <= 1:
            raise ValueError("s2_cs_threshold must be in [0, 1]")
        if resample not in (None, "bilinear", "bicubic"):
            raise ValueError("resample must be None, 'bilinear' or 'bicubic'")
        self.max_scene_cloud = max_scene_cloud
        self.s2_nir = s2_nir
        self.s2_cs_threshold = s2_cs_threshold
        self.mask_snow = mask_snow
        self.resample = resample

        if self.landsat_keys and self.s2_platforms is not None:
            warnings.warn(
                "Landsat and Sentinel-2 scenes are pooled into one stack without "
                "spectral band adjustment (no HLS-style harmonisation).",
                stacklevel=4,
            )

    def metadata(self) -> dict:
        sensors = list(self.landsat_keys)
        if self.s2_platforms is not None:
            sensors += self.s2_platforms or ["Sentinel-2 (all)"]
        return {
            "sensors": ",".join(sensors),
            "start_date": self.period.start.isoformat(),
            "end_date": (self.period.end - timedelta(days=1)).isoformat(),
        }

    # -- collections -------------------------------------------------------
    def _filter(self, coll, region, cloud_property):
        import ee

        coll = coll.filterBounds(region).filterDate(self.period.start.isoformat(),
                                                   self.period.end.isoformat())
        if self.max_scene_cloud is not None:
            coll = coll.filter(ee.Filter.lte(cloud_property, self.max_scene_cloud))
        return coll

    def _raw_landsat(self, key, region):
        import ee

        return self._filter(ee.ImageCollection(LANDSAT_COLLECTIONS[key][0]), region, "CLOUD_COVER")

    def _raw_s2(self, region):
        import ee

        s2 = self._filter(ee.ImageCollection(S2_COLLECTION), region, "CLOUDY_PIXEL_PERCENTAGE")
        if self.s2_platforms:
            s2 = s2.filter(ee.Filter.inList("SPACECRAFT_NAME", self.s2_platforms))
        return s2

    def _prep_landsat(self, image, src_bands):
        import ee

        # QA_PIXEL bits: 0 fill, 1 dilated cloud, 2 cirrus, 3 cloud, 4 shadow, 5 snow
        bits = 0b11111 | ((1 << 5) if self.mask_snow else 0)
        clear = image.select("QA_PIXEL").bitwiseAnd(bits).eq(0)
        unsaturated = image.select("QA_RADSAT").eq(0)
        optical = image.select(src_bands, COMMON_BANDS)
        if self.resample:
            optical = optical.resample(self.resample)  # masks stay nearest-neighbour
        optical = (optical.multiply(0.0000275).add(-0.2)
                   .updateMask(clear).updateMask(unsaturated))
        return ee.Image(optical.copyProperties(image, ["system:time_start"]))

    def _prep_s2(self, image):
        import ee

        clear = image.select("cs_cdf").gte(self.s2_cs_threshold)
        bad_scl = [0, 1] + ([11] if self.mask_snow else [])  # no data, defective, snow
        scl_ok = image.select("SCL").remap(bad_scl, [0] * len(bad_scl), 1)
        optical = image.select(["B2", "B3", "B4", self.s2_nir, "B11", "B12"], COMMON_BANDS)
        if self.resample:
            optical = optical.resample(self.resample)
        optical = optical.divide(10000).updateMask(clear).updateMask(scl_ok)
        return ee.Image(optical.copyProperties(image, ["system:time_start"]))

    def collection(self, region):
        """Masked, scaled, merged image collection (bands = ``self.bands``)."""
        import ee

        cols = []
        for key in self.landsat_keys:
            src_bands = LANDSAT_COLLECTIONS[key][1]
            cols.append(self._raw_landsat(key, region)
                        .map(lambda img, b=src_bands: self._prep_landsat(img, b)))
        if self.s2_platforms is not None:
            cols.append(
                self._raw_s2(region)
                .linkCollection(ee.ImageCollection(S2_CLOUD_SCORE), ["cs_cdf"])
                # keep only scenes for which a Cloud Score+ image exists
                .filter(ee.Filter.listContains("system:band_names", "cs_cdf"))
                .map(self._prep_s2)
            )
        merged = cols[0]
        for c in cols[1:]:
            merged = merged.merge(c)
        return merged.select(self.bands)

    def scene_counts(self, region) -> dict:
        """{sensor: number of scenes intersecting ``region`` in the period}."""
        import ee

        info = {}
        if self.landsat_keys:
            info["landsat"] = ee.Dictionary.fromLists(
                self.landsat_keys, [self._raw_landsat(k, region).size() for k in self.landsat_keys])
        if self.s2_platforms is not None:
            info["s2"] = self._raw_s2(region).aggregate_histogram("SPACECRAFT_NAME")
        info = ee.Dictionary(info).getInfo()
        return {**info.get("landsat", {}), **info.get("s2", {})}

    def prepare(self, grid, verbose=True):
        if not verbose:
            return
        counts = self.scene_counts(grid.ee_geometry())
        print("[eetool] scenes intersecting the extent: "
              + (", ".join(f"{k}={v}" for k, v in counts.items()) or "none"))
        missing = [p for p in (self.s2_platforms or []) if p not in counts]
        if missing:
            print(f"[eetool]   no scenes for: {', '.join(missing)}")

    @staticmethod
    def _empty(names):
        """Fully masked image with the given band names (-> NoData everywhere)."""
        import ee

        return ee.Image.constant([0] * len(names)).rename(names).updateMask(ee.Image.constant(0))


@register
class LandsatS2STM(_LandsatS2Base):
    """Cloud-masked per-pixel percentiles of Landsat and/or Sentinel-2 SR."""

    name = "landsat_s2_stm"

    def __init__(self, start_date, end_date, percentiles: Sequence[float] = (0.5,), **kwargs):
        self.percentiles = normalize_percentiles(percentiles)
        super().__init__(start_date, end_date, **kwargs)

    @property
    def band_names(self) -> list[str]:
        return [f"{percentile_label(p)}_{b}" for p in self.percentiles for b in self.bands]

    def metadata(self) -> dict:
        return {**super().metadata(),
                "percentiles": ",".join(f"{p:g}" for p in self.percentiles)}

    def build_image(self, region):
        import ee

        coll = self.collection(region)
        order = np.argsort(self.percentiles)
        reducer = ee.Reducer.percentile(
            [self.percentiles[i] * 100 for i in order],
            outputNames=[f"q{i}" for i in order],
        )
        out = [f"{b}_q{i}" for i in range(len(self.percentiles)) for b in self.bands]
        return ee.Image(ee.Algorithms.If(coll.size().gt(0),
                                         coll.reduce(reducer).select(out), self._empty(out)))


@register
class LandsatS2TE(_LandsatS2Base):
    """
    Temporal encoding: one cloud-masked composite of Landsat and/or
    Sentinel-2 SR per ``time_step`` days. Steps without any clear
    observation are NoData.
    """

    name = "landsat_s2_te"

    def __init__(self, start_date, end_date, time_step: int = 7, reducer: str = "mean",
                 **kwargs):
        super().__init__(start_date, end_date, **kwargs)
        if time_step is None or int(time_step) < 1:
            raise ValueError("time_step must be an integer >= 1 (days)")
        if reducer not in TE_REDUCERS:
            raise ValueError(f"reducer must be one of {TE_REDUCERS}")
        self.time_step = int(time_step)
        self.reducer = reducer
        self.windows = split_period(self.period, self.time_step)

    @property
    def dates(self) -> list[str]:
        """First day of every time step (ISO)."""
        return [w.start.isoformat() for w in self.windows]

    @property
    def band_names(self) -> list[str]:
        # time-major: all bands of step 0, then all bands of step 1, ...
        return [f"{b}_{d}" for d in self.dates for b in self.bands]

    def metadata(self) -> dict:
        return {**super().metadata(), "time_step_days": self.time_step,
                "reducer": self.reducer, "n_steps": len(self.windows),
                "band_order": "time-major"}

    def build_image(self, region):
        import ee

        coll = self.collection(region)
        steps = []
        for w, d in zip(self.windows, self.dates):
            sub = coll.filterDate(w.start.isoformat(), w.end.isoformat())
            names = [f"{b}_{d}" for b in self.bands]
            reduced = getattr(sub, self.reducer)().select(self.bands, names)
            steps.append(ee.Image(ee.Algorithms.If(sub.size().gt(0), reduced,
                                                   self._empty(names))))
        return ee.Image.cat(steps)

    def prepare(self, grid, verbose=True):
        super().prepare(grid, verbose)
        if verbose:
            last = self.windows[-1]
            print(f"[eetool] {len(self.windows)} time step(s) of {self.time_step} day(s) "
                  f"x {len(self.bands)} band(s); last step {last.start} .. "
                  f"{last.end - timedelta(days=1)} ({last.days} day(s))")


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------
_COMMON_DOC = """    sensors : sequence of str
        Any mix of ``"L4"``, ``"L5"``, ``"L7"``, ``"L8"``, ``"L9"`` and
        Sentinel-2 platforms ``"S2A"``, ``"S2B"``, ``"S2C"``, ... (``"S2"`` = all).
        All scenes are pooled into one stack.
    bands : sequence of str, optional
        Subset of ``COMMON_BANDS`` (blue, green, red, nir, swir1, swir2);
        default all six.
    max_scene_cloud : float, optional
        Drop whole scenes whose cloud cover (%) exceeds this before pixel masking.
    s2_nir : {"B8A", "B8"}
        Sentinel-2 band used as nir (B8A matches Landsat NIR better).
    s2_cs_threshold : float
        Cloud Score+ ``cs_cdf`` threshold (0.5-0.65 typical, higher = stricter).
    mask_snow : bool
        Mask snow/ice (Landsat QA bit 5, Sentinel-2 SCL class 11).
    resample : {None, "bilinear", "bicubic"}
        Resampling of optical bands when warping scenes to the grid
        (None = nearest neighbour).
    res, path_out, tile_size, n_workers, max_retries, verbose
        Download options, identical for every eetool function — see
        :func:`eetool.download`."""


def ee_download_landsat_s2_stm(
    raster_ref_image: GridLike,
    start_date,
    end_date,
    percentiles: Sequence[float] = (0.5,),
    sensors: Sequence[str] = ("L4", "L5", "L7", "L8", "L9"),
    bands: Sequence[str] | None = None,
    max_scene_cloud: float | None = None,
    s2_nir: str = "B8A",
    s2_cs_threshold: float = 0.6,
    mask_snow: bool = True,
    resample: str | None = None,
    *,
    res: float | None = None,
    path_out=None,
    tile_size: int | None = None,
    n_workers: int = 4,
    max_retries: int = 5,
    verbose: bool = True,
):
    product = LandsatS2STM(
        start_date, end_date, percentiles=percentiles, sensors=sensors, bands=bands,
        max_scene_cloud=max_scene_cloud, s2_nir=s2_nir, s2_cs_threshold=s2_cs_threshold,
        mask_snow=mask_snow, resample=resample,
    )
    return run(product, raster_ref_image, res=res, path_out=path_out, tile_size=tile_size,
               n_workers=n_workers, max_retries=max_retries, verbose=verbose)


ee_download_landsat_s2_stm.__doc__ = """
    Cloud-masked per-pixel percentile composite (spectral-temporal metrics) of
    Landsat and/or Sentinel-2 surface reflectance.

    Parameters
    ----------
    raster_ref_image : str | Path | eetool.Grid
        Reference raster (its upper-left corner, extent and CRS define the
        output grid) or a ready-made :class:`eetool.Grid`.
    start_date, end_date : str | datetime.date
        First and last day, BOTH INCLUSIVE: ``"20150101"`` / ``"2015-12-31"``.
    percentiles : sequence of float
        Fractions in [0, 1], e.g. ``[0.1, 0.5, 0.9]`` (0 / 1 = min / max).
        Output keeps the order given.
""" + _COMMON_DOC + """

    Returns
    -------
    RasterResult | str
        ``path_out is None`` -> :class:`eetool.RasterResult` with ``data`` of
        shape (n_percentiles * n_bands, H, W), int16 = reflectance * 10000,
        NoData = -9999, bands named ``"p10_blue"``, ``"p10_green"``, ...
        (percentile-major). Otherwise the GeoTIFF path.
"""


def ee_download_landsat_s2_te(
    raster_ref_image: GridLike,
    start_date,
    end_date,
    time_step: int = 7,
    sensors: Sequence[str] = ("L4", "L5", "L7", "L8", "L9"),
    bands: Sequence[str] | None = None,
    reducer: str = "mean",
    max_scene_cloud: float | None = None,
    s2_nir: str = "B8A",
    s2_cs_threshold: float = 0.6,
    mask_snow: bool = True,
    resample: str | None = None,
    *,
    res: float | None = None,
    path_out=None,
    tile_size: int | None = None,
    n_workers: int = 4,
    max_retries: int = 5,
    verbose: bool = True,
):
    product = LandsatS2TE(
        start_date, end_date, time_step=time_step, reducer=reducer, sensors=sensors,
        bands=bands, max_scene_cloud=max_scene_cloud, s2_nir=s2_nir,
        s2_cs_threshold=s2_cs_threshold, mask_snow=mask_snow, resample=resample,
    )
    return run(product, raster_ref_image, res=res, path_out=path_out, tile_size=tile_size,
               n_workers=n_workers, max_retries=max_retries, verbose=verbose)


ee_download_landsat_s2_te.__doc__ = """
    Temporal encoding: a regular time series of cloud-masked composites of
    Landsat and/or Sentinel-2 surface reflectance, one per ``time_step`` days.

    The period is cut into consecutive steps starting at ``start_date`` (the
    last step is cut at ``end_date`` and may be shorter). Within each step all
    clear observations are combined with ``reducer`` (default mean). Pixels
    with no clear observation in a step — clouds, shadow, snow, saturation,
    or no scene at all in that step — are NoData (-9999).

    Parameters
    ----------
    raster_ref_image : str | Path | eetool.Grid
        Reference raster (its upper-left corner, extent and CRS define the
        output grid) or a ready-made :class:`eetool.Grid`.
    start_date, end_date : str | datetime.date
        First and last day, BOTH INCLUSIVE: ``"20150101"`` / ``"2015-12-31"``.
    time_step : int
        Length of each step in days (e.g. 7 = weekly composites).
    reducer : {"mean", "median", "min", "max"}
        How clear observations within a step are combined.
""" + _COMMON_DOC + """

    Returns
    -------
    RasterResult | str
        ``path_out is None`` -> :class:`eetool.RasterResult` with ``data`` of
        shape (n_steps * n_bands, H, W), int16 = reflectance * 10000,
        NoData = -9999, bands named ``"blue_2015-01-01"``, ``"green_2015-01-01"``,
        ..., ``"blue_2015-01-08"``, ... (time-major, so
        ``data.reshape(n_steps, n_bands, H, W)`` gives a (T, C, H, W) cube).
        Otherwise the GeoTIFF path.
"""
