# %% Imports
from __future__ import annotations
import json
from typing import Iterable, Tuple, SupportsFloat as Numeric
import os
import requests
from tqdm import tqdm
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import pytz
import ephem
from dateutil.parser import parse
# %%
dict_dayofweek = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
dict_mon = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

def get_raw_tle_from_tstamp(ts: datetime | np.datetime64 | Iterable)->Tuple[datetime, str, str, str] | np.ndarray:
    if isinstance(ts, datetime):
        ts = datetime.utcfromtimestamp(ts.timestamp())
    elif isinstance(ts, np.datetime64):
        ts = datetime.utcfromtimestamp(int(ts)*1e-9)
    elif isinstance(ts, Iterable):
        out = []
        for t in tqdm(ts):
            out.append(get_raw_tle_from_tstamp(t))
        return np.asarray(out).T

    dayofweek = dict_dayofweek[ts.weekday()]
    day = ts.day
    mon = dict_mon[ts.month]
    year = ts.year
    hh = ts.hour
    mm = ts.minute
    ss = ts.second

    url = f'http://isstracker.com/ajax/fetchTLE.php?date={dayofweek}%2C%20{day}%20{mon}%20{year}%20{hh}%3A{mm}%3A{ss}%20GMT'

    content = requests.get(url)
    if content.status_code != 200:
        raise RuntimeError('Response %d'%(content.status_code))
    tledict = json.loads(content.content)
    epoch = tledict['epoch']
    lines = tledict['jsTLE'].replace('\r', '').split('\n')
    return (parse(epoch + '+00:00'), lines[0], lines[1], lines[2])
# %%
def staticvars(**kwargs):
    def decorate(func):
        for key in kwargs:
            setattr(func, key, kwargs[key])
        return func
    return decorate

@staticvars(tledb=None, tlefile='')
def ISSTleFromTstamp(ts: datetime | np.datetime64, *, database_fname: str = None, allowdownload: bool=True, full_output: bool=False) -> Tuple[str, str] | Tuple[str, str, datetime, bool, int]:
    """Get TLE for a given timestamp using ISS TLE database.

    Args:
        ts (datetime | np.datetime64): Timestamp for evaluation, must be timezone aware in case of datetime.
        database_fname (str, optional): TLE dataset file (loaded using xarray.load_dataset). The dataset file must contain a timestamp (coordinate) for when the TLE is valid, and data_vars line1 and line2 containing the two TLE lines. Defaults to 'ISS_TLE_DB.nc'.
        allowdownload (bool, optional): Allow download of TLE not found in DB.
        full_output (bool, optional): Return full output (line1, line2, epoch, found, idx). Defaults to False.

    Raises:
        ValueError: Timestamp must be timezone aware.
        IndexError: Could not find valid TLE in the dataset (allowdownload=False).
        RuntimeError: Could not download valid TLE (allowdownload=True, database does not contain valid epoch).

    Returns:
        Tuple[str, str] | Tuple[str, str, datetime, bool, int]: (line1, line2) or (line1, line2, datetime, found, idx) if full_output=True.
    """
    if ISSTleFromTstamp.tledb is None or ISSTleFromTstamp.tlefile != database_fname:
        if database_fname is None:
            database_fname = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ISS_TLE_DB.nc')
        ISSTleFromTstamp.tledb = xr.load_dataset(database_fname)
        ISSTleFromTstamp.tlefile = database_fname
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            raise ValueError('Timestamp must be timezone aware')
        ts = datetime.utcfromtimestamp(ts.astimezone(tz = pytz.utc).timestamp())
    elif isinstance(ts, np.datetime64):
        ts = datetime.utcfromtimestamp(int(ts)*1e-9)

    tledb: xr.Dataset = ISSTleFromTstamp.tledb

    tle_tstamps = tledb.timestamp.values.astype(int)*1e-9
    lows = np.diff(np.asarray(tle_tstamps < ts.timestamp(), dtype=int))

    if np.sum(lows) == 0: # no transitions => not found
        if allowdownload: # download is allowed
            _, _, l1, l2 = get_raw_tle_from_tstamp(ts) # get the TLE
            if full_output:
                return (l1, l2, ts, False, -1)
            else: return (l1, l2)
        else:
            raise IndexError('Could not find valid TLE.') # no can do
    else: # already in DB
        idx = np.where(lows != 0)[0]
        dts = tledb.timestamp.values[idx]
        tles = tledb.sel(dict(timestamp=dts))
        l1 = tles.line1.values[0]
        l2 = tles.line2.values[0]
        if full_output:
            return (l1, l2, dts[0], True, idx)
        else: return (l1, l2)
# %%
def ISSLatLonFromTstamp(ts: datetime | np.datetime64, *, database_fname: str = None, allowdownload: bool=True) -> Tuple[Numeric, Numeric, Numeric]:
    """Get latitude, longitude for a given timestamp using ISS TLE database.

    Args:
        ts (datetime | np.datetime64): Timestamp for evaluation, must be timezone aware in case of datetime.
        database_fname (str, optional): TLE dataset file (loaded using xarray.load_dataset). The dataset file must contain a timestamp (coordinate) for when the TLE is valid, and data_vars line1 and line2 containing the two TLE lines. Defaults to 'ISS_TLE_DB.nc'.
        allowdownload (bool, optional): Allow download of TLE not found in DB.

    Raises:
        ValueError: Timestamp must be timezone aware.
        IndexError: Could not find valid TLE in the dataset (allowdownload=False).
        RuntimeError: Could not download valid TLE (allowdownload=True, database does not contain valid epoch).

    Returns:
        Tuple[Numeric, Numeric, Numeric]: (latitude, longitude, altitude) in degrees (-180, 180) and km.
    """
    l1, l2 = ISSTleFromTstamp(ts, database_fname=database_fname, allowdownload=allowdownload)
    tle = ephem.readtle('GENERIC', l1, l2)
    tle.compute(ts)
    return (np.rad2deg(float(tle.sublat)), np.rad2deg(float(tle.sublong)), tle.elevation*1e-3)
# %%
if __name__ == '__main__':
    ts = pytz.timezone('US/Eastern').localize(datetime(2017, 4, 1, 0, 0, 1))
    print(ts, ISSTleFromTstamp(ts, full_output=True)[2:])
    print(ts, ISSLatLonFromTstamp(ts))
# %%
