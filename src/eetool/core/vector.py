"""
Extent and CRS of a vector file: ``vector_info(path) -> (crs, bounds, n)``.

Only the extent (xmin, ymin, xmax, ymax), the CRS and the feature count are
needed, so no geometry library is required for the two most common formats:

- GeoPackage (``.gpkg``) is an SQLite database: read with Python's own
  ``sqlite3`` (R-tree index, else the geometry blobs).
- Shapefile (``.shp``): the extent is in the ``.shp`` header, the CRS in the
  ``.prj`` file, the feature count in the ``.shx`` file.

Every other format (GeoJSON, FlatGeobuf, GeoParquet, ...) goes through the
first library that imports: pyogrio, fiona or GDAL's ``osgeo.ogr``. If a
library is installed but fails to import (e.g. a DLL conflict on Windows),
the real import error is reported.
"""

from __future__ import annotations

import math
import sqlite3
import struct
from pathlib import Path

#: (crs as WKT / "EPSG:xxxx" / None, (xmin, ymin, xmax, ymax) or None, n_features or -1)
VectorInfo = tuple


# --------------------------------------------------------------------------- #
# Built-in: GeoPackage
# --------------------------------------------------------------------------- #
def _gpkg_layer(con, layer):
    rows = con.execute("SELECT table_name, column_name, srs_id FROM gpkg_geometry_columns "
                       "ORDER BY table_name").fetchall()
    if not rows:
        raise ValueError("GeoPackage has no vector layers")
    if layer is None:
        return rows[0]
    if isinstance(layer, int):
        if not 0 <= layer < len(rows):
            raise ValueError(f"Layer index {layer} out of range (0..{len(rows) - 1})")
        return rows[layer]
    for row in rows:
        if row[0] == str(layer):
            return row
    raise ValueError(f"No layer {layer!r}; layers: {[r[0] for r in rows]}")


def _gpkg_crs(con, srs_id):
    if srs_id in (None, 0, -1):
        return None
    row = con.execute("SELECT organization, organization_coordsys_id, definition "
                      "FROM gpkg_spatial_ref_sys WHERE srs_id = ?", (srs_id,)).fetchone()
    if row is None:
        return None
    org, code, definition = row
    if org and str(org).upper() == "EPSG" and code not in (None, 0, -1):
        return f"EPSG:{int(code)}"
    if definition and definition.strip().lower() not in ("", "undefined"):
        return definition
    return None


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


# -- WKB parsing (only needed for geometries stored without an envelope) ----
def _wkb_coords(buf, offset=0):
    """Yield (x, y) of every vertex of the WKB geometry at ``offset``; returns end offset."""
    out = []

    def read(off):
        order = "<" if buf[off] == 1 else ">"
        (gtype,) = struct.unpack_from(order + "I", buf, off + 1)
        off += 5
        has_z = has_m = False
        if gtype & 0x80000000:  # EWKB flags
            has_z = True
        if gtype & 0x40000000:
            has_m = True
        if gtype & 0x20000000:  # EWKB SRID
            off += 4
        gtype &= 0x0FFFFFFF
        base, dim = gtype % 1000, gtype // 1000  # ISO: 1000 Z, 2000 M, 3000 ZM
        has_z = has_z or dim in (1, 3)
        has_m = has_m or dim in (2, 3)
        nd = 2 + has_z + has_m
        step = 8 * nd

        def points(off, n):
            for i in range(n):
                x, y = struct.unpack_from(order + "dd", buf, off + i * step)
                out.append((x, y))
            return off + n * step

        if base == 1:  # Point
            x, y = struct.unpack_from(order + "dd", buf, off)
            if not (math.isnan(x) or math.isnan(y)):  # empty point = NaN NaN
                out.append((x, y))
            return off + step
        if base in (2, 13, 8):  # LineString, Triangle-ish, CircularString: one ring
            (n,) = struct.unpack_from(order + "I", buf, off)
            return points(off + 4, n)
        if base in (3, 17):  # Polygon, Triangle
            (nrings,) = struct.unpack_from(order + "I", buf, off)
            off += 4
            for _ in range(nrings):
                (n,) = struct.unpack_from(order + "I", buf, off)
                off = points(off + 4, n)
            return off
        if base in (4, 5, 6, 7, 9, 10, 11, 12, 15, 16):  # Multi*, collections, curves
            (n,) = struct.unpack_from(order + "I", buf, off)
            off += 4
            for _ in range(n):
                off = read(off)
            return off
        raise ValueError(f"Unsupported WKB geometry type {gtype}")

    read(offset)
    return out


def gpkg_blob_bounds(blob):
    """(xmin, ymin, xmax, ymax) of one GeoPackage geometry blob, or None if empty."""
    if blob is None or len(blob) < 8 or blob[:2] != b"GP":
        return None
    flags = blob[3]
    order = "<" if flags & 1 else ">"
    envelope = (flags >> 1) & 0b111
    empty = (flags >> 4) & 1
    if empty:
        return None
    n_env = {0: 0, 1: 4, 2: 6, 3: 6, 4: 8}.get(envelope)
    if n_env is None:
        raise ValueError("Invalid GeoPackage geometry header")
    if n_env:
        minx, maxx, miny, maxy = struct.unpack_from(order + "dddd", blob, 8)
        return (minx, miny, maxx, maxy)
    coords = _wkb_coords(blob, 8 + 8 * n_env)
    if not coords:
        return None
    xs, ys = zip(*coords)
    return (min(xs), min(ys), max(xs), max(ys))


def _info_gpkg_builtin(path, layer=None, use_rtree=True) -> VectorInfo:
    uri = Path(path).resolve().as_uri() + "?mode=ro"
    con = sqlite3.connect(uri, uri=True)
    try:
        table, column, srs_id = _gpkg_layer(con, layer)
        crs = _gpkg_crs(con, srs_id)
        n = con.execute(f"SELECT count(*) FROM {_quote(table)}").fetchone()[0]
        if n == 0:
            return crs, None, 0
        bounds = None
        if use_rtree:
            try:
                row = con.execute(
                    f"SELECT min(minx), min(miny), max(maxx), max(maxy) "
                    f"FROM {_quote(f'rtree_{table}_{column}')}").fetchone()
                if row and None not in row:
                    bounds = tuple(float(v) for v in row)
            except sqlite3.Error:  # no spatial index, or no R-tree module
                bounds = None
        if bounds is None:
            xmin = ymin = math.inf
            xmax = ymax = -math.inf
            for (blob,) in con.execute(f"SELECT {_quote(column)} FROM {_quote(table)}"):
                b = gpkg_blob_bounds(blob)
                if b is None:
                    continue
                xmin, ymin = min(xmin, b[0]), min(ymin, b[1])
                xmax, ymax = max(xmax, b[2]), max(ymax, b[3])
            bounds = None if xmin == math.inf else (xmin, ymin, xmax, ymax)
        return crs, bounds, n
    finally:
        con.close()


# --------------------------------------------------------------------------- #
# Built-in: Shapefile
# --------------------------------------------------------------------------- #
def _sidecar(path: Path, ext: str):
    for e in (ext, ext.upper()):
        p = path.with_suffix(e)
        if p.exists():
            return p
    return None


def _info_shp_builtin(path, layer=None) -> VectorInfo:
    path = Path(path)
    with open(path, "rb") as f:
        header = f.read(100)
    if len(header) < 100 or struct.unpack(">i", header[:4])[0] != 9994:
        raise ValueError(f"{path} is not a valid shapefile")
    xmin, ymin, xmax, ymax = struct.unpack("<4d", header[36:68])
    prj = _sidecar(path, ".prj")
    crs = (prj.read_text(errors="replace").strip() or None) if prj else None
    shx = _sidecar(path, ".shx")
    if shx is not None:
        with open(shx, "rb") as f:
            (length_words,) = struct.unpack(">i", f.read(28)[24:28])
        n = (length_words * 2 - 100) // 8
    else:
        n = -1
    if n == 0 or not all(map(math.isfinite, (xmin, ymin, xmax, ymax))) or xmin > xmax:
        return crs, None, 0 if n == 0 else n
    return crs, (xmin, ymin, xmax, ymax), n


# --------------------------------------------------------------------------- #
# Libraries
# --------------------------------------------------------------------------- #
def _info_pyogrio(path, layer) -> VectorInfo:
    import pyogrio

    info = pyogrio.read_info(path, layer=layer, force_total_bounds=True)
    return info.get("crs") or None, info.get("total_bounds"), info.get("features", -1)


def _info_fiona(path, layer) -> VectorInfo:
    import fiona

    with fiona.open(path, layer=layer) as src:
        crs = src.crs.to_wkt() if hasattr(src.crs, "to_wkt") else (src.crs_wkt or None)
        return crs or None, tuple(src.bounds), len(src)


def _info_ogr(path, layer) -> VectorInfo:
    from osgeo import ogr

    ds = ogr.Open(path)
    if ds is None:
        raise ValueError(f"GDAL/OGR cannot open {path}")
    lyr = (ds.GetLayer(0) if layer is None else
           ds.GetLayer(layer) if isinstance(layer, int) else ds.GetLayerByName(str(layer)))
    if lyr is None:
        raise ValueError(f"No layer {layer!r} in {path}")
    srs = lyr.GetSpatialRef()
    n = lyr.GetFeatureCount()
    if not n:
        return (srs.ExportToWkt() if srs is not None else None), None, 0
    xmin, xmax, ymin, ymax = lyr.GetExtent(True)
    return (srs.ExportToWkt() if srs is not None else None), (xmin, ymin, xmax, ymax), n


#: tried in this order for formats without a built-in reader
LIBRARY_READERS = (("pyogrio", _info_pyogrio), ("fiona", _info_fiona),
                   ("osgeo", _info_ogr))
BUILTIN_READERS = {".gpkg": _info_gpkg_builtin, ".shp": _info_shp_builtin}


def vector_info(path, layer=None) -> VectorInfo:
    """
    ``(crs, (xmin, ymin, xmax, ymax), n_features)`` of a vector file.
    ``crs`` is a WKT / ``"EPSG:xxxx"`` string or None; bounds are None when
    the layer has no geometries; ``n_features`` is -1 when unknown.
    """
    path = str(path)
    builtin = BUILTIN_READERS.get(Path(path).suffix.lower())
    if builtin is not None:
        return builtin(path, layer)

    problems = []
    for name, reader in LIBRARY_READERS:
        try:
            return reader(path, layer)
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.split(".")[0] != name:
                problems.append(f"  {name}: installed, but importing it failed: {exc}")
            else:
                problems.append(f"  {name}: not installed")
        except ImportError as exc:  # installed but broken (e.g. DLL load failed)
            problems.append(f"  {name}: installed, but importing it failed: {exc}")
    raise ImportError(
        f"Cannot read the vector file {path}: GeoPackage (.gpkg) and Shapefile (.shp) "
        "work without extra libraries, other formats need pyogrio, fiona or GDAL's "
        "osgeo.\n" + "\n".join(problems) + "\n"
        "Fix with one of:\n"
        "    conda install -c conda-forge pyogrio     (in a conda environment)\n"
        "    pip install pyogrio\n"
        "or convert the file to .gpkg."
    )
