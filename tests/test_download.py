"""Download engine and file layouts, tested offline: computePixels is faked."""

import numpy as np
import pytest
import rasterio

import ee
import eetool
from eetool.core import Encoding, Product, run

TIMES = ["2024-01-01", "2024-01-08", "2024-01-15"]
CHANNELS = ["a", "b"]


class FakeProduct(Product):
    """3 times x 2 channels; value = 1000 * time + 100 * channel + row (a) / col (b)."""

    name = "fake"
    default_res = 30
    encoding = Encoding.scaled(1, "int16", -9999)

    @property
    def times(self):
        return TIMES

    @property
    def channels(self):
        return CHANNELS

    def build_image(self, region):  # pragma: no cover - not used offline
        raise NotImplementedError

    def request_image(self, grid, tile, band_ids):
        return {"tile": tile, "ids": band_ids}


def expected():
    rr, cc = np.mgrid[0:70, 0:100]
    out = np.empty((3, 70, 100, 2), np.int16)
    for t in range(3):
        out[t, ..., 0] = 1000 * t + rr
        out[t, ..., 1] = 1000 * t + 100 + cc
    return out


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
        ids = request["bandIds"]
        out = np.zeros((h, w), dtype=[(b, "i2") for b in ids])
        for i, b in enumerate(ids):           # time-major band order
            t, c = divmod(i, 2)
            out[b] = 1000 * t + (rr if c == 0 else 100 + cc)
        calls.append(request)
        return out

    monkeypatch.setattr(ee.data, "computePixels", compute)
    return calls


def test_in_memory_is_4d(ref_tif, fake_compute):
    res = run(FakeProduct(), ref_tif, tile_size=32, verbose=False)
    assert isinstance(res, eetool.RasterResult)
    assert res.shape == (3, 70, 100, 2)            # (time, H, W, channel)
    assert res.times == TIMES and res.channels == CHANNELS
    assert np.array_equal(res.data, expected())
    assert np.asarray(res).shape == (3, 70, 100, 2)
    assert len(fake_compute) == 3 * 4


def test_sel_and_to_bands(ref_tif, fake_compute):
    res = run(FakeProduct(), ref_tif, verbose=False)
    sub = res.sel(time="2024-01-08", channel=["b"])
    assert sub.shape == (1, 70, 100, 1) and sub.times == ["2024-01-08"]
    assert np.array_equal(sub.data[0, ..., 0], expected()[1, ..., 1])
    bands = res.to_bands()
    assert bands.shape == (6, 70, 100)
    assert np.array_equal(bands[3], expected()[1, ..., 1])   # time 1, channel b
    assert res.band_names[3] == "b_2024-01-08"
    with pytest.raises(KeyError):
        res.sel(channel="zzz")


def test_stream_stack(ref_tif, fake_compute, tmp_path):
    out = run(FakeProduct(), ref_tif, path_out=tmp_path / "x" / "o.tif",
              tile_size=32, n_workers=3, verbose=False)
    assert out == str(tmp_path / "x" / "o.tif")
    with rasterio.open(out) as src:
        assert src.count == 6
        assert src.descriptions[:3] == ("a_2024-01-01", "b_2024-01-01", "a_2024-01-08")
        assert src.nodata == -9999 and src.crs.to_epsg() == 32633
        assert src.tags()["eetool_product"] == "fake"
    back = eetool.read_raster(out)
    assert np.array_equal(back.data, expected())
    assert back.times == TIMES and back.channels == CHANNELS and back.product == "fake"


def test_stream_per_time(ref_tif, fake_compute, tmp_path):
    out = run(FakeProduct(), ref_tif, path_out=tmp_path / "s1.tif", layout="per_time",
              tile_size=32, verbose=False)
    assert [p.rsplit("/", 1)[1] for p in out] == [
        "s1_2024-01-01.tif", "s1_2024-01-08.tif", "s1_2024-01-15.tif"]
    with rasterio.open(out[1]) as src:
        assert src.count == 2 and src.descriptions == ("a", "b")
    one = eetool.read_raster(out[2])
    assert one.shape == (1, 70, 100, 2) and one.times == ["2024-01-15"]
    back = eetool.read_raster(out)
    assert np.array_equal(back.data, expected()) and back.times == TIMES


def test_result_to_raster_matches_stream(ref_tif, fake_compute, tmp_path):
    res = run(FakeProduct(), ref_tif, verbose=False)
    path = res.to_raster(tmp_path / "r.tif")
    with rasterio.open(path) as src:
        assert np.array_equal(src.read(), res.to_bands())
        assert src.transform == res.transform
    paths = res.to_raster(tmp_path / "r.tif", layout="per_time")
    assert len(paths) == 3
    with pytest.raises(ValueError):
        res.to_raster(tmp_path / "r.tif", layout="zarr")


def test_read_plain_geotiff(ref_tif):
    res = eetool.read_raster(ref_tif)
    assert res.shape == (1, 70, 100, 1)


def test_to_xarray(ref_tif, fake_compute):
    pytest.importorskip("xarray")
    da = run(FakeProduct(), ref_tif, verbose=False).to_xarray()
    assert da.dims == ("time", "y", "x", "channel")
    assert list(da.time.values) == TIMES
    assert float(da.x[0]) == 500015 and float(da.y[0]) == 4999985


def test_failure_leaves_no_file(ref_tif, monkeypatch, tmp_path):
    def boom(request):
        raise ee.EEException("Image.select: Pattern 'x' did not match any bands.")

    monkeypatch.setattr(ee.data, "computePixels", boom)
    for layout in ("stack", "per_time"):
        with pytest.raises(RuntimeError, match="did not match"):
            run(FakeProduct(), ref_tif, path_out=tmp_path / "o.tif", layout=layout,
                verbose=False)
    assert list(tmp_path.glob("o*")) == []


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
    with pytest.raises(ValueError):
        run(FakeProduct(), "ref.tif", layout="netcdf")
