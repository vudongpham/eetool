"""
Concrete datasets. Importing this package registers every product.

To add a dataset, create a module here with a ``@register``-ed
:class:`eetool.core.Product` subclass plus an ``ee_get_<name>`` wrapper,
then import it below and in ``eetool/__init__.py``.
"""

from .embedding import AnnualEmbedding, dequantize_embedding, ee_get_annual_embedding, quantize_embedding
from .landsat_s2 import (COMMON_BANDS, LandsatS2STM, LandsatS2TE, ee_get_landsat_s2_stm,
                         ee_get_landsat_s2_te)
from .sentinel1 import S1Composite, dequantize_s1, ee_get_s1_composite, quantize_s1

__all__ = [
    "AnnualEmbedding", "COMMON_BANDS", "LandsatS2STM", "LandsatS2TE", "S1Composite",
    "dequantize_embedding", "dequantize_s1", "ee_get_annual_embedding",
    "ee_get_landsat_s2_stm", "ee_get_landsat_s2_te", "ee_get_s1_composite", "quantize_embedding",
    "quantize_s1",
]
