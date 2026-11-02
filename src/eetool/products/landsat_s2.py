"""
Spectral-temporal metrics (per-pixel percentile composites) of Landsat
4/5/7/8/9 Collection 2 Level-2 and/or Sentinel-2 L2A (harmonized; any
platform A, B, C, D, ...) surface reflectance.

Storage: reflectance * 10000 as int16, NoData = -9999.
"""

from __future__ import annotations

import re
import warnings
from typing import Sequence

import numpy as np

from ..core import Encoding, Product, register, resolve_period, run
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


def _parse_months(months):
    if months is None:
        return None
    months = sorted({int(m) for m in ([months] if isinstance(months, int) else months)})
    if not months or months[0] < 1 or months[-1] > 12:
        raise ValueError("months must be integers in 1..12")
    return months


# ---------------------------------------------------------------------------
# Product
# ---------------------------------------------------------------------------
@register
class LandsatS2STM(Product):
    """Cloud-masked per-pixel percentiles of Landsat and/or Sentinel-2 SR."""

    name = "landsat_s2_stm"
    default_res = 30
    encoding = ENCODING

    def __init__(
        self,
        year: int | None = None,
        start_date=None,
        end_date=None,
        percentiles: Sequence[float] = (0.5,),
        sensors: Sequence[str] = ("L4", "L5", "L7", "L8", "L9"),
        bands: Sequence[str] | None = None,
        months: Sequence[int] | None = None,
        max_scene_cloud: float | None = None,
        s2_nir: str = "B8A",
        s2_cs_threshold: float = 0.6,
        mask_snow: bool = True,
        resample: str | None = None,
    ):
        self.period = resolve_period(year, start_date, end_date)
        self.percentiles = normalize_percentiles(percentiles)
        self.landsat_keys, self.s2_platforms = parse_sensors(sensors)
        self.bands = _parse_bands(bands)
        self.months = _parse_months(months)
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
                stacklevel=3,
            )

    @property
    def band_names(self) -> list[str]:
        return [f"{percentile_label(p)}_{b}" for p in self.percentiles for b in self.bands]

    def metadata(self) -> dict:
        sensors = list(self.landsat_keys)
        if self.s2_platforms is not None:
            sensors += self.s2_platforms or ["Sentinel-2 (all)"]
        return {
            "sensors": ",".join(sensors),
            "start_date": self.period.start.isoformat(),
            "end_date_exclusive": self.period.end.isoformat(),
            "percentiles": ",".join(f"{p:g}" for p in self.percentiles),
            "months": ",".join(map(str, self.months)) if self.months else "all",
        }

    # -- collections -------------------------------------------------------
    def _filter(self, coll, region, cloud_property):
        import ee

        coll = coll.filterBounds(region).filterDate(self.period.start.isoformat(),
                                                   self.period.end.isoformat())
        if self.months:
            coll = coll.filter(ee.Filter.Or(*[ee.Filter.calendarRange(m, m, "month")
                                              for m in self.months]))
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

    # -- product interface -------------------------------------------------
    def build_image(self, region):
        import ee

        coll = self.collection(region)
        order = np.argsort(self.percentiles)
        reducer = ee.Reducer.percentile(
            [self.percentiles[i] * 100 for i in order],
            outputNames=[f"q{i}" for i in order],
        )
        out = [f"{b}_q{i}" for i in range(len(self.percentiles)) for b in self.bands]
        empty = (ee.Image.constant([0] * len(out)).rename(out)
                 .updateMask(ee.Image.constant(0)))
        return ee.Image(ee.Algorithms.If(coll.size().gt(0),
                                         coll.reduce(reducer).select(out), empty))

    def prepare(self, grid, verbose=True):
        if not verbose:
            return
        import ee

        region = grid.ee_geometry()
        info = {}
        if self.landsat_keys:
            info["landsat"] = ee.Dictionary.fromLists(
                self.landsat_keys, [self._raw_landsat(k, region).size() for k in self.landsat_keys])
        if self.s2_platforms is not None:
            info["s2"] = self._raw_s2(region).aggregate_histogram("SPACECRAFT_NAME")
        info = ee.Dictionary(info).getInfo()
        counts = {**info.get("landsat", {}), **info.get("s2", {})}
        print("[eetool] scenes intersecting the extent: "
              + (", ".join(f"{k}={v}" for k, v in counts.items()) or "none"))
        missing = [p for p in (self.s2_platforms or []) if p not in info.get("s2", {})]
        if missing:
            print(f"[eetool]   no scenes for: {', '.join(missing)}")


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------
def ee_download_landsat_s2_stm(
    raster_ref_image: GridLike,
    year: int | None = None,
    start_date=None,
    end_date=None,
    percentiles: Sequence[float] = (0.5,),
    sensors: Sequence[str] = ("L4", "L5", "L7", "L8", "L9"),
    bands: Sequence[str] | None = None,
    months: Sequence[int] | None = None,
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
    """
    Cloud-masked per-pixel percentile composite (spectral-temporal metrics) of
    Landsat and/or Sentinel-2 surface reflectance.

    Parameters
    ----------
    raster_ref_image : str | Path | eetool.Grid
        Reference raster (its upper-left corner, extent and CRS define the
        output grid) or a ready-made :class:`eetool.Grid`.
    year : int, optional
        Shortcut for ``start_date=f"{year}-01-01", end_date=f"{year+1}-01-01"``.
    start_date, end_date : str | datetime.date, optional
        ISO dates; ``end_date`` is EXCLUSIVE.
    percentiles : sequence of float
        Fractions in [0, 1], e.g. ``[0.1, 0.5, 0.9]`` (0 / 1 = min / max).
        Output keeps the order given.
    sensors : sequence of str
        Any mix of ``"L4"``, ``"L5"``, ``"L7"``, ``"L8"``, ``"L9"`` and
        Sentinel-2 platforms ``"S2A"``, ``"S2B"``, ``"S2C"``, ... (``"S2"`` = all).
        All scenes are pooled into one stack.
    bands : sequence of str, optional
        Subset of ``COMMON_BANDS`` (blue, green, red, nir, swir1, swir2);
        default all six.
    months : sequence of int, optional
        Keep only scenes acquired in these calendar months (e.g. ``[6, 7, 8]``).
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
        :func:`eetool.download`.

    Returns
    -------
    RasterResult | str
        ``path_out is None`` -> :class:`eetool.RasterResult` with ``data`` of
        shape (n_percentiles * n_bands, H, W), int16 = reflectance * 10000,
        NoData = -9999, bands named ``"p10_blue"``, ``"p10_green"``, ...
        (percentile-major). Otherwise the GeoTIFF path.
    """
    product = LandsatS2STM(
        year=year, start_date=start_date, end_date=end_date, percentiles=percentiles,
        sensors=sensors, bands=bands, months=months,
        max_scene_cloud=max_scene_cloud, s2_nir=s2_nir, s2_cs_threshold=s2_cs_threshold,
        mask_snow=mask_snow, resample=resample,
    )
    return run(product, raster_ref_image, res=res, path_out=path_out, tile_size=tile_size,
               n_workers=n_workers, max_retries=max_retries, verbose=verbose)
