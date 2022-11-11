from __future__ import annotations
from typing import List
import datetime as dt
import numpy as np
import pytz
from astropy.time import Time
import astropy.coordinates as coord

# Source: https://stackoverflow.com/questions/23699115/datetime-with-pytz-timezone-different-offset-depending-on-how-tzinfo-is-set


def datetime_in_timezone(tzinfo: pytz.BaseTzInfo, year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0, microsecond: int = 0) -> dt.datetime:
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

def get_localtime(time: List[float] | np.ndarray | float | dt.datetime, lat: List[float] | float, lon: List[float] | float)->float | np.ndarray:
    """Get local time (solar) for a given timestamp at a given geolocation.

    Args:
        time (List[float] | np.ndarray | float | dt.datetime): Timestamp (must be in np.datetime64 format for ndarray, datetime.timestamp() format otherwise.)
        lat (List[float]): Latitude (-90 deg to 90 deg)
        lon (List[float]): Longitude (0 deg to 360 deg)

    Raises:
        ValueError: Time, latitude, longitude arrays are of different length.

    Returns:
        float | np.ndarray: Evaluated local time (0 is midnight, 12 is noon).
    """
    timeisfloat = False
    if isinstance(time, list):
        pass
    elif isinstance(time, np.ndarray):
        time = time.tolist()
    elif isinstance(time, float):
        timeisfloat = True
        time = np.asarray([dt.datetime.fromtimestamp(t) for t in time], dtype=np.datetime64)  # now it is in datetime64 from float timestamp
    elif isinstance(time, dt.datetime):
        timeisfloat = True
        time = [np.datetime64(time)]  # now it is in datetime64
    if isinstance(lon, float):
        lon = [lon]
    if isinstance(lat, float):
        lat = [lat]
    if len(time) != len(lon) != len(lat):
        raise ValueError('Lists are not of equal length')
    t = Time(time, format='datetime64', scale='utc')
    scoords = coord.get_sun(t)  # geodesic earth coords. ra,dec in degrees
    tt = Time(t, format='datetime64', scale='utc', location=(
        lon, lat))  # time on the ISS with location of ISS
    lst = [x.sidereal_time(kind='apparent')
           for x in tt]  # local sidereal time in degrees
    # local time noon = 0
    lt = [(lst[i] - scoords[i].ra).hourangle for i in range(len(lst))]
    lt = np.asarray(lt, dtype='float')
    lt -= 12  # offset
    lt[np.where(lt < 0)] += 24  # wrap around
    if not timeisfloat:
        return (lt)
    return float(lt)
