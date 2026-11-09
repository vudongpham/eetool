"""
eetool — Google Earth Engine data as numpy arrays on your own pixel grid.

Every function returns a :class:`RasterResult` whose ``data`` is a 4-D array
(time, height, width, channel), or streams it to GeoTIFF with ``path_out``.

    import eetool
    from eetool import (ee_get_landsat_s2_stm, ee_get_landsat_s2_te,
                        ee_get_annual_embedding, ee_get_s1_composite)

    eetool.initialize(project="my-gcp-project")
    stm = ee_get_landsat_s2_stm("ref.tif", "20240101", "20241231",
                                     percentiles=[0.1, 0.5, 0.9], sensors=["L8", "L9"])
    stm.data.shape            # (3, H, W, 6): (percentile, H, W, band)
    stm.to_raster("stm_2024.tif")
"""

from __future__ import annotations

from .core import (
    DOWNLOAD_OPTIONS,
    Encoding,
    LAYOUTS,
    Grid,
    Product,
    RasterResult,
    download,
    get,
    get_product,
    list_products,
    read_raster,
    register,
    write_raster,
)
from .products import (
    COMMON_BANDS,
    S2_BANDS,
    AnnualEmbedding,
    LandsatS2STM,
    LandsatS2TE,
    S1Composite,
    dequantize_embedding,
    dequantize_s1,
    ee_get_annual_embedding,
    ee_get_landsat_s2_stm,
    ee_get_landsat_s2_te,
    ee_get_s1_composite,
    quantize_embedding,
    quantize_s1,
)

__version__ = "0.3.0"

HIGH_VOLUME_URL = "https://earthengine-highvolume.googleapis.com"


def initialize(project: str | None = None, high_volume: bool = True, **kwargs) -> None:
    """
    ``ee.Initialize`` with the high-volume endpoint (recommended for the
    parallel ``computePixels`` requests eetool makes). Extra kwargs go to
    ``ee.Initialize``.
    """
    import ee

    if high_volume:
        kwargs.setdefault("opt_url", HIGH_VOLUME_URL)
    ee.Initialize(project=project, **kwargs)


__all__ = [
    "COMMON_BANDS", "DOWNLOAD_OPTIONS", "S2_BANDS", "LAYOUTS", "AnnualEmbedding", "Encoding", "Grid",
    "LandsatS2STM", "LandsatS2TE", "Product", "RasterResult", "S1Composite", "__version__",
    "dequantize_embedding", "dequantize_s1", "download", "get", "ee_get_annual_embedding",
    "ee_get_landsat_s2_stm", "ee_get_landsat_s2_te", "ee_get_s1_composite", "get_product", "initialize",
    "list_products", "quantize_embedding", "read_raster", "quantize_s1", "register", "write_raster",
]
