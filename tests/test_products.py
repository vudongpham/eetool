import pytest

import eetool
from eetool.products.landsat_s2 import LandsatS2STM, LandsatS2TE, parse_sensors, percentile_label
from eetool.products.sentinel1 import S1Composite


def test_registry():
    assert {"landsat_s2_stm", "landsat_s2_te", "annual_embedding", "s1_composite"} <= set(eetool.list_products())


def test_public_imports():
    from eetool import (ee_get_annual_embedding, ee_get_landsat_s2_stm,  # noqa: F401
                        ee_get_landsat_s2_te, ee_get_s1_composite)


def test_sensor_parsing():
    assert parse_sensors(["landsat-8", "L9"]) == (["L8", "L9"], None)
    assert parse_sensors("S2") == ([], [])
    assert parse_sensors(["Sentinel-2C", "s2a"]) == ([], ["Sentinel-2C", "Sentinel-2A"])
    with pytest.raises(ValueError):
        parse_sensors(["L6"])


def test_stm_band_names():
    p = LandsatS2STM("20240101", "20241231", percentiles=[0.9, 0.025], sensors=["L8"],
                     bands=["red", "nir"])
    assert percentile_label(0.025) == "p2.5"
    assert p.times == ["p90", "p2.5"]       # one time step per percentile
    assert p.channels == ["red", "nir"]
    assert p.shape == (2, 2)
    assert p.band_names == ["red_p90", "nir_p90", "red_p2.5", "nir_p2.5"]
    one = LandsatS2STM("20240101", "20241231", percentiles=[0.5], sensors=["L8"])
    assert one.shape == (1, 6) and one.band_names == one.channels
    assert p.encoding.dtype == "int16"


def test_stm_validation():
    with pytest.raises(ValueError):
        LandsatS2STM("20240101", "20241231", percentiles=[50])
    with pytest.raises(ValueError):
        LandsatS2STM("20240101", "20241231", bands=["foo"])
    with pytest.warns(UserWarning):
        LandsatS2STM("20240101", "20241231", sensors=["L8", "S2"])


def test_s1_band_names():
    p = S1Composite("20240101", "20241231", polarizations=["vv", "VH"], time_step=100, orbit="desc")
    assert p.orbit == "DESCENDING"
    assert p.times == ["2024-01-01", "2024-04-10", "2024-07-19", "2024-10-27"]
    assert p.channels == ["VV", "VH"]
    assert p.shape == (4, 2)
    assert p.band_names[:3] == ["VV_2024-01-01", "VH_2024-01-01", "VV_2024-04-10"]
    assert S1Composite("20240101", "20241231", platforms=["S1A", "Sentinel-1C", "b"]).platforms == ["A", "C", "B"]
    with pytest.raises(ValueError):
        S1Composite("20240101", "20241231", polarizations=["XX"])


def test_embedding_bands():
    p = eetool.AnnualEmbedding(2024)
    assert p.shape == (1, 64) and p.channels[0] == "A00" and p.times == ["2024"]
    assert p.encoding.nodata == 65535


def test_te_steps_and_band_names():
    p = LandsatS2TE(start_date="20150101", end_date="20151231", time_step=7,
                    sensors=["L8", "L9"], bands=["red", "nir"])
    assert p.shape == (53, 2)              # 52 full weeks + 31 Dec
    assert p.windows[-1].days == 1
    assert p.times[:2] == ["2015-01-01", "2015-01-08"] and p.times[-1] == "2015-12-31"
    assert p.channels == ["red", "nir"]
    assert p.encoding.nodata == -9999
    with pytest.raises(ValueError):
        LandsatS2TE("20150101", "20151231", time_step=0)
    with pytest.raises(ValueError):
        LandsatS2TE("20150101", "20151231", reducer="p90")


def test_te_nodata_out():
    import numpy as np

    assert LandsatS2TE("20150101", "20150131").encoding.nodata == -9999
    p = LandsatS2TE("20150101", "20150131", NoDataOut=-32768, sensors=["L8"])
    enc = p.encoding
    assert enc.nodata == -32768 and enc.dtype == "int16"
    q = enc.encode([0.1234, np.nan, -10.0])           # -10 would clamp onto -32768
    assert q.tolist() == [1234, -32768, -32767]
    assert np.isnan(enc.decode(q)[1]) and np.isclose(enc.decode(q)[0], 0.1234)
    assert LandsatS2TE("20150101", "20150131", NoDataOut=-20000.0).encoding.nodata == -20000
    with pytest.warns(UserWarning, match="inside the range"):
        LandsatS2TE("20150101", "20150131", NoDataOut=0)
    for bad in (40000, 1.5, float("nan")):
        with pytest.raises(ValueError):
            LandsatS2TE("20150101", "20150131", NoDataOut=bad)
    with pytest.raises(TypeError):
        LandsatS2TE("20150101", "20150131", NoDataOut="x")


def test_s2_only_gets_all_ten_bands():
    from eetool import S2_BANDS

    for sensors in ("S2", ["S2A", "S2B"], ("S2",)):
        p = LandsatS2TE("20250101", "20251231", time_step=30, sensors=sensors)
        assert p.channels == S2_BANDS
        assert S2_BANDS == ["blue", "green", "red", "re1", "re2", "re3", "bnir", "nir",
                            "swir1", "swir2"]
    stm = LandsatS2STM("20250101", "20251231", percentiles=[0.5], sensors="S2",
                       bands=["RE1", "bnir", "nir"])
    assert stm.channels == ["re1", "bnir", "nir"]


def test_landsat_keeps_six_bands_and_rejects_s2_only_bands():
    assert LandsatS2TE("20250101", "20251231", sensors=["L8"]).channels == COMMON_BANDS6
    with pytest.warns(UserWarning):              # Landsat + S2 pooled
        mixed = LandsatS2STM("20250101", "20251231", sensors=["L8", "S2"])
    assert mixed.channels == COMMON_BANDS6
    for sensors in (["L8"], ["L9", "S2"]):
        with pytest.raises(ValueError, match="only for Sentinel-2"):
            LandsatS2TE("20250101", "20251231", sensors=sensors, bands=["red", "re1"])
        with pytest.raises(ValueError, match="only for Sentinel-2"):
            LandsatS2STM("20250101", "20251231", sensors=sensors, bands=["bnir"])
    with pytest.raises(ValueError, match="Unknown band"):
        LandsatS2TE("20250101", "20251231", sensors="S2", bands=["re4"])


COMMON_BANDS6 = ["blue", "green", "red", "nir", "swir1", "swir2"]


def test_public_te_per_time_files(ref_tif, tmp_path, monkeypatch):
    """ee_get_landsat_s2_te(..., path_out=..., layout="per_time") end to end
    (Earth Engine faked): one GeoTIFF per 30-day step with the 10 S2 bands."""
    import ee
    import numpy as np
    import rasterio

    import eetool

    monkeypatch.setattr(LandsatS2TE, "request_image", lambda self, g, t, ids: None)

    def compute(request):
        d = request["grid"]["dimensions"]
        return np.ones((d["height"], d["width"]), dtype=[(b, "i2") for b in request["bandIds"]])

    monkeypatch.setattr(ee.data, "computePixels", compute)
    paths = eetool.ee_get_landsat_s2_te(
        ref_tif, start_date="2025-01-01", end_date="2025-12-31", time_step=30,
        sensors=("S2"), path_out=tmp_path / "te.tif", layout="per_time")
    assert len(paths) == 13 and paths[0].endswith("te_2025-01-01.tif")
    with rasterio.open(paths[-1]) as src:
        assert src.count == 10 and src.descriptions[3] == "re1"
    back = eetool.read_raster(paths)
    assert back.shape == (13, 70, 100, 10)
    with pytest.raises(TypeError, match="per_time"):
        eetool.ee_get_landsat_s2_te(ref_tif, "2025-01-01", "2025-12-31", sensors="S2",
                                    per_time="times")


def test_s1_uses_time_step():
    p = S1Composite("20240101", "20241231", time_step=30)
    assert len(p.times) == 13 and p.times[1] == "2024-01-31"
    assert S1Composite("20240101", "20241231", time_step=None).times == ["2024-01-01"]
    with pytest.raises(TypeError):
        S1Composite("20240101", "20241231", window_days=10)


def test_s1_readme_db_conversions(tmp_path, ref_tif):
    """The four ways in the README give the same dB values."""
    import numpy as np
    import rasterio

    import eetool
    from eetool.core import Grid, RasterResult
    from eetool.products.sentinel1 import s1_encoding

    db_true = np.array([[-40.0, -15.3], [0.0, np.nan]], dtype="float32")
    enc = s1_encoding()
    q = enc.encode(db_true)[None, ..., None]                  # (1, 2, 2, 1)
    grid = Grid.from_bounds((0, 0, 20, 20), "EPSG:32633", 10)
    res = RasterResult(q, ["2024-01-01"], ["VV"], grid, enc, "s1_composite")

    def s1_to_db(q, db_min=-40.0, db_max=10.0):
        q = np.asarray(q)
        db = q.astype("float32") * (db_max - db_min) / 65534 + db_min
        return np.where(q == 65535, np.nan, db)

    path = res.to_raster(tmp_path / "s1.tif")
    with rasterio.open(path) as src:
        qq = src.read()
        factor = float(src.tags()["eetool_factor"])
        offset = float(src.tags()["eetool_offset"])
        from_file = np.where(qq == src.nodata, np.nan, qq / factor + offset)

    ways = [res.decode()[0, ..., 0], eetool.dequantize_s1(res.data)[0, ..., 0],
            s1_to_db(res.data)[0, ..., 0], from_file[0],
            eetool.read_raster(path).decode()[0, ..., 0]]
    for db in ways:
        assert np.allclose(db, db_true, atol=0.001, equal_nan=True)
