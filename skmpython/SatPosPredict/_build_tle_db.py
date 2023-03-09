# %% Imports
from __future__ import annotations
from typing import Iterable, List, Tuple
from tqdm import tqdm
import xarray as xr
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import pytz
from dateutil.parser import parse

import json
import requests

# %%
def datetime_from_julian(t: str) -> datetime:
    year = int(t[:2]) # first two digits
    if year > 56: # first launch in 57 so 57 is 1957
        year += 1900
    else: # < 56: 56 -> 2056
        year += 2000

    yday = int(t[2:5]) # day of year
    fday = float(t[5:]) # fractional day of year
    
    start = pytz.utc.localize(datetime(year, 1, 1)) # first day of the year
    tstamp = start.timestamp()
    tstamp += (yday - 1)*86400 # add seconds spent per day
    tstamp += fday*86400 # fraction of day to seconds
    return datetime.utcfromtimestamp(tstamp)

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
def get_tle_ds(tledb: np.ndarray):
    if not isinstance(tledb, np.ndarray) or tledb.shape[0] != 4:
        raise RuntimeError
    tstamp = np.asarray(tledb[0, :], dtype=np.datetime64)
    line_0 = np.asarray(tledb[1, :], dtype=str)
    line_1 = np.asarray(tledb[2, :], dtype=str)
    line_2 = np.asarray(tledb[3, :], dtype=str)

    tle_ds = xr.Dataset(
    data_vars={'line0': ('timestamp', line_0),
        'line1': ('timestamp', line_1), 'line2': ('timestamp', line_2)},
    coords={'timestamp': tstamp}
    )
    return tle_ds.drop_duplicates(dim=..., keep='first')

# %%
start = pytz.utc.localize(datetime(2017, 3, 29))
tstamps = []
cond = True
idx = 0
while cond:
    next = datetime.utcfromtimestamp(start.timestamp() + 15*60*idx)
    idx += 1
    tstamps.append(next)
    if next > pytz.utc.localize(datetime(2017, 6, 1)).replace(tzinfo=None):
        break
# %%
out = get_raw_tle_from_tstamp(tstamps)
# %%
ds = get_tle_ds(out)
# %%
ds.to_netcdf('ISS_TLE_DB.nc')
# %%
