"""Download engine tested offline: computePixels is faked."""

import numpy as np
import pytest
import rasterio

import ee
import eetool
from eetool.core import Encoding, Product, run


class FakeProduct(Product):
    name = "fake"
    default_res = 30
    encoding = Encoding.scaled(1, "int16", -9999)

    @property
    def band_names(self):
        return ["a", "b"]

    def build_image(self, region):  # pragma: no cover - not used offline
        raise NotImplementedError

    def request_image(self, grid, tile, band_ids):
        return {"tile": tile, "ids": band_ids}


@pytest.fixture
def fake_compute(monkeypatch):
    calls = []

    def compute(request):
        g = request["grid"]
        w, h = g["dimensions"]["width"], g["dimensions"]["height"]
        x0 = g["affineTransform"]["translateX"]
        y1 = g["affineTransform"]["translateY"]
        res = g["affineTransform"]["scaleX"]
        cols = (x0 - 500000) / res + np.arange(w)
        rows = (5000000 - y1) / res + np.arange(h)
        rr, cc = np.meshgrid(rows, cols, indexing="ij")
        out = np.zeros((h, w), dtype=[(b, "i2") for b in request["bandIds"]])
        out[request["bandIds"][0]] = rr
        out[request["bandIds"][1]] = cc
        calls.append(request)
        return out

    monkeypatch.setattr(ee.data, "computePixels", compute)
    return calls


def test_in_memory(ref_tif, fake_compute):
    res = run(FakeProduct(), ref_tif, tile_size=32, verbose=False)
    assert isinstance(res, eetool.RasterResult)
    assert res.shape == (2, 70, 100)
    rr, cc = np.mgrid[0:70, 0:100]
    assert np.array_equal(res.band("a"), rr) and np.array_equal(res.band("b"), cc)
    assert len(fake_compute) == 3 * 4


def test_stream_to_disk(ref_tif, fake_compute, tmp_path):
    out = run(FakeProduct(), ref_tif, path_out=tmp_path / "x" / "o.tif",
              tile_size=32, n_workers=3, verbose=False)
    with rasterio.open(out) as src:
        assert src.descriptions == ("a", "b")
        assert src.nodata == -9999 and src.crs.to_epsg() == 32633
        assert np.array_equal(src.read(2), np.mgrid[0:70, 0:100][1])
        assert src.tags()["eetool_product"] == "fake"


def test_result_to_raster_matches(ref_tif, fake_compute, tmp_path):
    res = run(FakeProduct(), ref_tif, verbose=False)
    path = res.to_raster(tmp_path / "r.tif")
    with rasterio.open(path) as src:
        assert np.array_equal(src.read(), res.data)
        assert src.transform == res.grid.transform


def test_failure_leaves_no_file(ref_tif, monkeypatch, tmp_path):
    def boom(request):
        raise ee.EEException("Image.select: Pattern 'x' did not match any bands.")

    monkeypatch.setattr(ee.data, "computePixels", boom)
    with pytest.raises(RuntimeError, match="did not match"):
        run(FakeProduct(), ref_tif, path_out=tmp_path / "o.tif", verbose=False)
    assert list(tmp_path.glob("o.tif*")) == []


def test_retry_transient(ref_tif, monkeypatch):
    state = {"n": 0}

    def flaky(request):
        state["n"] += 1
        if state["n"] == 1:
            raise ee.EEException("Too many concurrent aggregations.")
        h = request["grid"]["dimensions"]["height"]
        w = request["grid"]["dimensions"]["width"]
        return np.ones((h, w), dtype=[(b, "i2") for b in request["bandIds"]])

    monkeypatch.setattr(ee.data, "computePixels", flaky)
    monkeypatch.setattr("time.sleep", lambda s: None)
    res = run(FakeProduct(), ref_tif, n_workers=1, verbose=False)
    assert (res.data == 1).all()


def test_tile_size_capped():
    from eetool.core.download import _safe_tile_size
    with pytest.warns(UserWarning):
        assert _safe_tile_size(4096, 64, 2) <= 572


def test_download_by_name_kwargs_split():
    with pytest.raises(TypeError):
        eetool.download(FakeProduct(), "ref.tif", start_date="20240101")
