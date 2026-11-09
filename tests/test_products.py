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
    p = S1Composite("20240101", "20241231", polarizations=["vv", "VH"], window_days=100, orbit="desc")
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
