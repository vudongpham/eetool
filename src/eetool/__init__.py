"""
eetool — download Google Earth Engine data onto your own pixel grid.

    import eetool
    from eetool import (ee_download_landsat_s2_stm, ee_download_annual_embedding,
                        ee_download_s1_composite)

    eetool.initialize(project="my-gcp-project")
    stm = ee_download_landsat_s2_stm("ref.tif", year=2024, percentiles=[0.1, 0.5, 0.9],
                                     sensors=["L8", "L9"])
    stm.to_raster("stm_2024.tif")
"""

from __future__ import annotations

from .core import (
    DOWNLOAD_OPTIONS,
    Encoding,
    Grid,
    Product,
    RasterResult,
    download,
    get_product,
    list_products,
    register,
    write_raster,
)
from .products import (
    COMMON_BANDS,
    AnnualEmbedding,
    LandsatS2STM,
    S1Composite,
    dequantize_embedding,
    dequantize_s1,
    ee_download_annual_embedding,
    ee_download_landsat_s2_stm,
    ee_download_s1_composite,
    quantize_embedding,
    quantize_s1,
)

__version__ = "0.1.0"

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
    "COMMON_BANDS", "DOWNLOAD_OPTIONS", "AnnualEmbedding", "Encoding", "Grid",
    "LandsatS2STM", "Product", "RasterResult", "S1Composite", "__version__",
    "dequantize_embedding", "dequantize_s1", "download", "ee_download_annual_embedding",
    "ee_download_landsat_s2_stm", "ee_download_s1_composite", "get_product", "initialize",
    "list_products", "quantize_embedding", "quantize_s1", "register", "write_raster",
]
