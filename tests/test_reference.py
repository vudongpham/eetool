"""Reference inputs: raster files, vector files, GeoDataFrames."""

import pytest

from eetool.core import Grid


gpd = pytest.importorskip("geopandas")
shapely = pytest.importorskip("shapely")

from eetool.core.grid import reference_kind, resolve_grid  # noqa: E402


@pytest.fixture
def aoi():
    return gpd.GeoDataFrame(geometry=[shapely.box(500100, 4990050, 505000, 4994990),
                                      shapely.Point(509990, 4999990)], crs="EPSG:32633")


def test_vector_gpkg_snapped(aoi, tmp_path):
    path = tmp_path / "aoi.gpkg"
    aoi.to_file(path)
    assert reference_kind(path) == "vector"
    g = resolve_grid(path, 30)
    assert g.crs.to_epsg() == 32633
    assert g.bounds == (500100, 4990050, 510000, 5000010)   # snapped out to 30 m
    assert (g.width, g.height) == (330, 332)
    assert g.x_ul % 30 == 0 and g.y_ul % 30 == 0


def test_vector_geodataframe_and_explicit_crs(aoi):
    g = Grid.from_reference(aoi, 10)
    assert g.bounds == (500100, 4990050, 509990, 4999990)   # already multiples of 10
    g2 = Grid.from_vector(aoi, 0.001, crs="EPSG:4326")
    assert g2.crs.to_epsg() == 4326 and 15 < g2.x_ul < 15.1   # UTM 33 centre = 15°E


def test_vector_geographic_goes_to_utm(aoi, tmp_path):
    path = tmp_path / "aoi_wgs84.shp"
    aoi.set_geometry(aoi.buffer(5)).to_crs(4326).to_file(path)   # polygons only
    with pytest.warns(UserWarning, match="EPSG:32633"):
        g = resolve_grid(path, 30)
    assert g.crs.to_epsg() == 32633
    left, bottom, right, top = g.bounds
    assert left <= 500100 and right >= 509990 and bottom <= 4990050 and top >= 4999990


def test_vector_needs_res(aoi, tmp_path):
    with pytest.raises(ValueError, match="res"):
        Grid.from_vector(aoi, None)
    with pytest.raises(TypeError):
        resolve_grid(123, 30)


def test_raster_reference_still_works(ref_tif):
    assert reference_kind(ref_tif) == "raster"
    assert resolve_grid(ref_tif, None).res == 30


def test_run_with_vector_reference(aoi, tmp_path, monkeypatch):
    import ee
    import numpy as np

    from eetool.core import Encoding, Product, run

    class One(Product):
        name = "one"
        default_res = 100
        encoding = Encoding.scaled(1, "int16", -9999)
        times = ["t0", "t1"]
        channels = ["a"]

        def build_image(self, region):  # pragma: no cover
            raise NotImplementedError

        def request_image(self, grid, tile, band_ids):
            return None

    def compute(request):
        d = request["grid"]["dimensions"]
        return np.ones((d["height"], d["width"]), dtype=[(b, "i2") for b in request["bandIds"]])

    monkeypatch.setattr(ee.data, "computePixels", compute)
    path = tmp_path / "aoi.gpkg"
    aoi.to_file(path)
    res = run(One(), path, verbose=False)            # res=None -> product default 100 m
    assert res.shape == (2, 100, 99, 1) and res.grid.res == 100   # (time, H, W, channel)
    assert res.transform.c == 500100 and res.transform.f == 5000000
    assert (res.data == 1).all()
