from datetime import date

import numpy as np
import pytest

from eetool.core import Encoding, Grid, resolve_period, split_period


# --- encoding ----------------------------------------------------------------
def test_from_range_roundtrip():
    enc = Encoding.from_range(-1, 1, "uint16")
    x = np.array([-1.0, -0.5, 0.0, 0.123456, 1.0, np.nan])
    q = enc.encode(x)
    assert q.dtype == np.uint16
    assert q[0] == 0 and q[4] == 65534 and q[5] == 65535
    back = enc.decode(q)
    assert np.isnan(back[5])
    assert np.allclose(back[:5], x[:5], atol=enc.step / 2 + 1e-9)


def test_from_range_clips():
    enc = Encoding.from_range(-40, 10)
    assert enc.encode([-100, 100]).tolist() == [0, 65534]


def test_embedding_encoding_matches_original_formula():
    enc = Encoding.from_range(-1, 1, "uint16")
    x = np.linspace(-1, 1, 1001)
    original = np.rint((x + 1.0) * 0.5 * 65534).astype(np.uint16)
    assert np.array_equal(enc.encode(x), original)


def test_scaled_int16():
    enc = Encoding.scaled(10000, "int16", -9999)
    assert enc.encode([0.1234, np.nan]).tolist() == [1234, -9999]
    assert np.isclose(enc.decode(np.array([1234]))[0], 0.1234)


# --- grid --------------------------------------------------------------------
def test_grid_from_raster(ref_tif):
    g = Grid.from_raster(ref_tif, 10)
    assert (g.width, g.height) == (300, 210)
    assert g.ee_crs == "EPSG:32633"
    g30 = Grid.from_raster(ref_tif)  # native resolution
    assert (g30.width, g30.height, g30.res) == (100, 70, 30)
    g_odd = Grid.from_raster(ref_tif, 40)  # extent rounded up
    assert (g_odd.width, g_odd.height) == (75, 53)


def test_tiles_cover_grid_exactly(ref_tif):
    g = Grid.from_raster(ref_tif, 10)
    cover = np.zeros(g.shape, int)
    for t in g.tiles(64):
        cover[t.row:t.row + t.height, t.col:t.col + t.width] += 1
    assert (cover == 1).all()


def test_tile_request(ref_tif):
    g = Grid.from_raster(ref_tif, 30)
    t = g.tiles(32)[1]
    req = g.tile_grid_request(t)
    assert req["affineTransform"]["translateX"] == 500000 + 32 * 30
    assert req["affineTransform"]["translateY"] == 5000000
    assert req["crsCode"] == "EPSG:32633"


def test_geographic_with_metres_rejected():
    with pytest.raises(ValueError):
        Grid.from_bounds((10, 50, 11, 51), "EPSG:4326", 30)


# --- dates -------------------------------------------------------------------
def test_periods():
    p = resolve_period("20240101", "20241231")       # end date inclusive
    assert p == (date(2024, 1, 1), date(2025, 1, 1))
    assert resolve_period("2024-01-01", date(2024, 12, 31)) == p
    assert resolve_period("20240305", "20240305").days == 1
    w = split_period(p, 10)
    assert len(w) == 37 and w[-1].end == date(2025, 1, 1) and w[-1].days == 6
    assert split_period(p, None) == [p]
    with pytest.raises(ValueError):
        resolve_period("20240201", "20240131")
    with pytest.raises(ValueError):
        resolve_period("2024/01/01", "20241231")
    with pytest.raises(ValueError):
        resolve_period("20240101", None)

