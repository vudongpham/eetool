"""
Linear value encodings used to store physical values as compact integers.

Every product describes how its float values are stored with one
:class:`Encoding`::

    q = round((x - offset) * factor), clamped to [qmin, qmax]
    x = q / factor + offset
    NoData = nodata  (never produced by a valid value when qmin/qmax exclude it)

The same encoding is applied on the Earth Engine side (``encode_ee``) before
download, so only integers travel over the wire, and on the client side
(``encode`` / ``decode``) for round trips.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Earth Engine cast method for each supported dtype.
_EE_CAST = {
    "uint8": "toUint8",
    "int8": "toInt8",
    "uint16": "toUint16",
    "int16": "toInt16",
    "uint32": "toUint32",
    "int32": "toInt32",
    "float32": "toFloat",
}


@dataclass(frozen=True)
class Encoding:
    """Linear float <-> integer mapping with a reserved NoData value."""

    dtype: str
    nodata: float
    factor: float = 1.0
    offset: float = 0.0
    qmin: float | None = None
    qmax: float | None = None

    def __post_init__(self):
        if self.dtype not in _EE_CAST:
            raise ValueError(f"Unsupported dtype {self.dtype!r}; use one of {sorted(_EE_CAST)}")
        if self.factor == 0:
            raise ValueError("factor must be non-zero")
        if self.is_integer:
            info = np.iinfo(self.dtype)
            object.__setattr__(self, "qmin", info.min if self.qmin is None else self.qmin)
            object.__setattr__(self, "qmax", info.max if self.qmax is None else self.qmax)

    # ------------------------------------------------------------------ #
    # Constructors
    # ------------------------------------------------------------------ #
    @classmethod
    def from_range(cls, vmin: float, vmax: float, dtype: str = "uint16") -> "Encoding":
        """
        Map the physical range [vmin, vmax] onto the full unsigned integer
        range, keeping the dtype's largest value as NoData.

        e.g. uint16: vmin -> 0, vmax -> 65534, NoData = 65535.
        """
        if not vmax > vmin:
            raise ValueError("vmax must be greater than vmin")
        info = np.iinfo(dtype)
        if info.min != 0:
            raise ValueError("from_range expects an unsigned dtype (uint8/uint16/uint32)")
        qmax = info.max - 1
        return cls(dtype=dtype, nodata=info.max, factor=qmax / (vmax - vmin),
                   offset=vmin, qmin=0, qmax=qmax)

    @classmethod
    def scaled(cls, factor: float, dtype: str = "int16", nodata: float = -9999) -> "Encoding":
        """Plain scaling, e.g. reflectance * 10000 stored as int16."""
        return cls(dtype=dtype, nodata=nodata, factor=factor, offset=0.0)

    @classmethod
    def float32(cls, nodata: float = -9999.0) -> "Encoding":
        """No quantisation: values are downloaded as float32."""
        return cls(dtype="float32", nodata=nodata)

    # ------------------------------------------------------------------ #
    @property
    def is_integer(self) -> bool:
        return np.issubdtype(np.dtype(self.dtype), np.integer)

    @property
    def step(self) -> float:
        """Physical size of one quantisation step (max error = step / 2)."""
        return 1.0 / abs(self.factor)

    # ------------------------------------------------------------------ #
    # Client side
    # ------------------------------------------------------------------ #
    def encode(self, x) -> np.ndarray:
        """float array (NaN = NoData) -> encoded array."""
        x = np.asarray(x, dtype=np.float64)
        q = (x - self.offset) * self.factor
        if self.is_integer:
            q = np.clip(np.rint(q), self.qmin, self.qmax)
        q = np.where(np.isnan(x), self.nodata, q)
        return q.astype(self.dtype)

    def decode(self, q) -> np.ndarray:
        """encoded array -> float32 physical values, NoData -> NaN."""
        q = np.asarray(q)
        x = q.astype(np.float64) / self.factor + self.offset
        x[q == self.nodata] = np.nan
        return x.astype(np.float32)

    # ------------------------------------------------------------------ #
    # Earth Engine side
    # ------------------------------------------------------------------ #
    def encode_ee(self, image):
        """Same encoding on the server; masked pixels become NoData."""
        if self.offset:
            image = image.subtract(self.offset)
        if self.factor != 1:
            image = image.multiply(self.factor)
        if self.is_integer:
            image = image.round().clamp(self.qmin, self.qmax)
        image = image.unmask(self.nodata, False)
        return getattr(image, _EE_CAST[self.dtype])()
