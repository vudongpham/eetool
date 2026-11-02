import numpy as np
import pytest
import rasterio
from affine import Affine


@pytest.fixture
def ref_tif(tmp_path):
    """100 x 70 px reference raster at 30 m in UTM 33N."""
    path = tmp_path / "ref.tif"
    with rasterio.open(path, "w", driver="GTiff", width=100, height=70, count=1,
                       dtype="uint8", crs="EPSG:32633",
                       transform=Affine(30, 0, 500000, 0, -30, 5000000)) as dst:
        dst.write(np.zeros((1, 70, 100), np.uint8))
    return path
