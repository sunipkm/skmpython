from __future__ import annotations
import datetime as dt
import pytz

# Source: https://stackoverflow.com/questions/23699115/datetime-with-pytz-timezone-different-offset-depending-on-how-tzinfo-is-set
def datetime_in_timezone(tzinfo: pytz.BaseTzInfo, year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0, microsecond: int = 0)->dt.datetime:
    """Create datetime object with a specified time zone.

    Args:
        tzinfo (pytz.BaseTzInfo): Time zone info. (e.g. pytz.timezone('US/Eastern'))
        year (int): Year
        month (int): Month (1-12)
        day (int): Day of month (1-31)
        hour (int, optional): Hour of day. Defaults to 0.
        minute (int, optional): Minute of day. Defaults to 0.
        second (int, optional): Second of day. Defaults to 0.
        microsecond (int, optional): Microsecond of day. Defaults to 0.

    Returns:
        dt.datetime: Datetime object for the specified time zone.
    """
    return tzinfo.localize(dt.datetime(year, month, day, hour, minute, second, microsecond))