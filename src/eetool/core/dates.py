"""Date helpers shared by the temporal products."""

from __future__ import annotations

from datetime import date, timedelta
from typing import NamedTuple


class Period(NamedTuple):
    """Half-open date interval [start, end) — same as ``ee.ImageCollection.filterDate``."""

    start: date
    end: date

    @property
    def days(self) -> int:
        return (self.end - self.start).days


def _to_date(value) -> date:
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def resolve_period(year=None, start_date=None, end_date=None) -> Period:
    """
    Either ``year`` (-> Jan 1 .. Jan 1 of the next year) or both
    ``start_date`` and ``end_date`` (ISO strings or ``datetime.date``;
    ``end_date`` is EXCLUSIVE).
    """
    if year is not None:
        if start_date is not None or end_date is not None:
            raise ValueError("Pass either `year` or `start_date`/`end_date`, not both")
        year = int(year)
        return Period(date(year, 1, 1), date(year + 1, 1, 1))
    if start_date is None or end_date is None:
        raise ValueError("Pass `year`, or both `start_date` and `end_date`")
    start, end = _to_date(start_date), _to_date(end_date)
    if end <= start:
        raise ValueError("end_date must be later than start_date (end_date is exclusive)")
    return Period(start, end)


def split_period(period: Period, window_days: int | None) -> list[Period]:
    """
    Consecutive non-overlapping windows of ``window_days`` days; the last one
    is cut at ``period.end``. ``None`` -> a single window (the whole period).
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
