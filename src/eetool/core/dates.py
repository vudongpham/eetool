"""Date helpers shared by the temporal products."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import NamedTuple


class Period(NamedTuple):
    """Half-open date interval [start, end) — same as ``ee.ImageCollection.filterDate``."""

    start: date
    end: date

    @property
    def days(self) -> int:
        return (self.end - self.start).days


def parse_date(value) -> date:
    """``datetime.date``, ``"YYYYMMDD"`` or ``"YYYY-MM-DD"`` -> ``date``."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    try:
        if len(text) == 8 and text.isdigit():
            return datetime.strptime(text, "%Y%m%d").date()
        return date.fromisoformat(text)
    except ValueError:
        raise ValueError(f"Invalid date {value!r}; use 'YYYYMMDD' or 'YYYY-MM-DD'") from None


def resolve_period(start_date, end_date) -> Period:
    """
    ``start_date`` .. ``end_date``, BOTH INCLUSIVE (``"20150101"``, ``"2015-12-31"``
    or ``datetime.date``). Returned as the half-open :class:`Period`
    ``[start_date, end_date + 1 day)``.
    """
    if start_date is None or end_date is None:
        raise ValueError("Pass both `start_date` and `end_date`, e.g. '20150101', '20151231'")
    start, last = parse_date(start_date), parse_date(end_date)
    if last < start:
        raise ValueError(f"end_date ({last}) is before start_date ({start})")
    return Period(start, last + timedelta(days=1))


def split_period(period: Period, window_days: int | None) -> list[Period]:
    """
    Consecutive non-overlapping windows of ``window_days`` days starting at
    ``period.start``; the last one is cut at ``period.end`` (so it may be
    shorter). ``None`` -> a single window (the whole period).
    """
    if window_days is None:
        return [period]
    window_days = int(window_days)
    if window_days < 1:
        raise ValueError("window_days must be >= 1")
    out, start = [], period.start
    while start < period.end:
        end = min(start + timedelta(days=window_days), period.end)
        out.append(Period(start, end))
        start = end
    return out
