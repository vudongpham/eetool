import pytest

import eetool
from eetool.products.landsat_s2 import LandsatS2STM, parse_sensors, percentile_label
from eetool.products.sentinel1 import S1Composite


def test_registry():
    assert {"landsat_s2_stm", "annual_embedding", "s1_composite"} <= set(eetool.list_products())


def test_public_imports():
    from eetool import (ee_download_annual_embedding, ee_download_landsat_s2_stm,  # noqa: F401
                        ee_download_s1_composite)


def test_sensor_parsing():
    assert parse_sensors(["landsat-8", "L9"]) == (["L8", "L9"], None)
    assert parse_sensors("S2") == ([], [])
    assert parse_sensors(["Sentinel-2C", "s2a"]) == ([], ["Sentinel-2C", "Sentinel-2A"])
    with pytest.raises(ValueError):
        parse_sensors(["L6"])


def test_stm_band_names():
    p = LandsatS2STM(year=2024, percentiles=[0.9, 0.025], sensors=["L8"],
                     bands=["red", "nir"])
    assert percentile_label(0.025) == "p2.5"
    assert p.band_names == ["p90_red", "p90_nir", "p2.5_red", "p2.5_nir"]
    assert p.encoding.dtype == "int16"


def test_stm_validation():
    with pytest.raises(ValueError):
        LandsatS2STM(year=2024, percentiles=[50])
    with pytest.raises(ValueError):
        LandsatS2STM(year=2024, bands=["foo"])
    with pytest.warns(UserWarning):
        LandsatS2STM(year=2024, sensors=["L8", "S2"])


def test_s1_band_names():
    p = S1Composite(year=2024, polarizations=["vv", "VH"], window_days=100, orbit="desc")
    assert p.orbit == "DESCENDING"
    assert p.band_names == ["VV_2024-01-01", "VV_2024-04-10", "VV_2024-07-19", "VV_2024-10-27",
                            "VH_2024-01-01", "VH_2024-04-10", "VH_2024-07-19", "VH_2024-10-27"]
    assert S1Composite(year=2024, platforms=["S1A", "Sentinel-1C", "b"]).platforms == ["A", "C", "B"]
    with pytest.raises(ValueError):
        S1Composite(year=2024, polarizations=["XX"])


def test_embedding_bands():
    p = eetool.AnnualEmbedding(2024)
    assert len(p.band_names) == 64 and p.band_names[0] == "A00"
    assert p.encoding.nodata == 65535
