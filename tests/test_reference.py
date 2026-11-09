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


def test_vector_gpkg_clipped_to_extent(aoi, tmp_path):
    path = tmp_path / "aoi.gpkg"
    aoi.to_file(path)
    assert reference_kind(path) == "vector"
    g = resolve_grid(path, 30)
    assert g.crs.to_epsg() == 32633
    # extent: xmin=500100, ymin=4990050, xmax=509990, ymax=4999990
    assert (g.x_ul, g.y_ul) == (500100, 4999990)            # starts exactly at the extent
    assert (g.width, g.height) == (330, 332)                # ceil(9890/30), ceil(9940/30)
    left, bottom, right, top = g.bounds
    assert 509990 <= right < 509990 + 30 and 4990050 - 30 < bottom <= 4990050
    snapped = Grid.from_vector(path, 30, snap=True)
    assert snapped.bounds == (500100, 4990050, 510000, 5000010)


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
    res = run(One(), path, showProgress=False)            # res=None -> product default 100 m
    assert res.shape == (2, 100, 99, 1) and res.grid.res == 100   # (time, H, W, channel)
    assert res.transform.c == 500100 and res.transform.f == 4999990
    assert (res.data == 1).all()


def _block_imports(monkeypatch, *names):
    """Make ``import name`` raise ImportError for each name."""
    import builtins
    import sys

    real_import = builtins.__import__
    for n in names:
        monkeypatch.delitem(sys.modules, n, raising=False)

    def fake_import(name, *args, **kwargs):
        if name.split(".")[0] in names:
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_vector_falls_back_to_fiona(aoi, tmp_path, monkeypatch):
    pytest.importorskip("fiona")
    path = tmp_path / "aoi.gpkg"
    aoi.to_file(path)
    expected = resolve_grid(path, 30)
    _block_imports(monkeypatch, "pyogrio")
    assert resolve_grid(path, 30) == expected


def test_gpkg_and_shp_need_no_library(aoi, tmp_path, monkeypatch):
    gpkg, shp = tmp_path / "aoi.gpkg", tmp_path / "aoi.shp"
    aoi.to_file(gpkg)
    polys = aoi.set_geometry(aoi.buffer(5))
    polys.to_file(shp)
    expected_gpkg, expected_shp = resolve_grid(gpkg, 30), resolve_grid(shp, 30)
    _block_imports(monkeypatch, "pyogrio", "fiona", "osgeo")
    assert resolve_grid(gpkg, 30) == expected_gpkg
    assert resolve_grid(shp, 30) == expected_shp
    assert expected_shp.crs.to_epsg() == 32633          # CRS read from the .prj


def test_other_formats_report_why(aoi, tmp_path, monkeypatch):
    path = tmp_path / "aoi.geojson"
    aoi.to_file(path)
    _block_imports(monkeypatch, "pyogrio", "fiona", "osgeo")
    with pytest.raises(ImportError, match="pyogrio: not installed") as err:
        resolve_grid(path, 30)
    assert "conda install -c conda-forge pyogrio" in str(err.value)
    assert Grid.from_reference(aoi, 30).res == 30     # GeoDataFrames need no reader


def test_broken_library_import_is_reported(aoi, tmp_path, monkeypatch):
    import builtins

    path = tmp_path / "aoi.geojson"
    aoi.to_file(path)
    _block_imports(monkeypatch, "fiona", "osgeo")
    real_import = builtins.__import__

    def dll_error(name, *args, **kwargs):
        if name == "pyogrio":
            raise ImportError("DLL load failed while importing _io: The specified "
                              "procedure could not be found.")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", dll_error)
    with pytest.raises(ImportError, match="pyogrio: installed, but importing it failed: "
                                          "DLL load failed"):
        resolve_grid(path, 30)


def test_gpkg_builtin_matches_pyogrio(tmp_path):
    pyogrio = pytest.importorskip("pyogrio")
    from eetool.core.vector import _info_gpkg_builtin

    gdf = gpd.GeoDataFrame(
        geometry=[shapely.Point(3, 4), shapely.LineString([(-1, 2), (5, 9)]),
                  shapely.Polygon([(0, 0), (10, 0), (10, 1)]),
                  shapely.MultiPoint([(20, -5), (21, 30)])],
        crs="EPSG:3035")
    path = tmp_path / "mixed.gpkg"
    gdf.to_file(path, layer="mixed")
    gdf.iloc[:1].to_file(path, layer="other", mode="a")
    ref = pyogrio.read_info(path, layer="mixed", force_total_bounds=True)
    crs, bounds, n = _info_gpkg_builtin(path, "mixed")
    assert crs == "EPSG:3035" and n == 4
    assert bounds == pytest.approx(tuple(ref["total_bounds"]))
    assert _info_gpkg_builtin(path, "other")[1] == (3, 4, 3, 4)
    with pytest.raises(ValueError, match="layers"):
        _info_gpkg_builtin(path, "nope")


def test_gpkg_extent_is_exact_not_rtree(tmp_path, monkeypatch):
    """A 30 km EPSG:3035 tile with float noise (like a FORCE / QGIS grid tile)
    must give exactly 1000 x 1000 pixels at 30 m; the R-tree's float32 values
    are ~0.5 m too large and used to give 1001 x 1001."""
    x0, y0 = 4526026.363052994, 3404919.607708866
    x1, y1 = 4556026.363054758, 3434919.607714256      # 30 km + ~2e-6 m noise
    tile = gpd.GeoDataFrame(geometry=[shapely.box(x0, y0, x1, y1)], crs="EPSG:3035")
    path = tmp_path / "tile.gpkg"
    tile.to_file(path)
    _block_imports(monkeypatch, "pyogrio", "fiona", "osgeo")   # built-in reader
    g = resolve_grid(path, 30)
    assert (g.height, g.width) == (1000, 1000)
    assert (g.x_ul, g.y_ul) == (x0, y1)                         # exact corner
    assert resolve_grid(path, 10).shape == (3000, 3000)
    assert resolve_grid(path, 20).shape == (1500, 1500)


def test_pixel_count_tolerance():
    from eetool.core.grid import n_pixels

    assert n_pixels(30000.0, 30) == 1000
    assert n_pixels(30000.00001, 30) == 1000           # float noise
    assert n_pixels(30000.02, 30) == 1000              # < 0.001 px over
    assert n_pixels(30000.1, 30) == 1001               # a real partial pixel
    assert n_pixels(29999.9, 30) == 1000
    assert n_pixels(1e-9, 30) == 1                      # never zero
